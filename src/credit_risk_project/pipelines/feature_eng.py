from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocess_raw_data, 
    identity_node, 
    split_data, 
    apply_categorical_xgb, 
    scale_data_ann, 
    smote_balance
)

def create_fen_model_pipeline(model_name: str) -> Pipeline:
    """Factory for model-specific feature engineering using a configuration mapping."""
    
    # Model Configuration Mapping
    FEN_CONFIG = {
        "catboost": {
            "pre_split": identity_node,
            "post_split": "identity",
            "outputs": ["train_ready_catboost", "test_split_catboost"]
        },
        "xgboost": {
            "pre_split": apply_categorical_xgb,
            "post_split": "identity",
            "outputs": ["train_ready_xgboost", "test_split_xgboost"]
        },
        "ann": {
            "pre_split": identity_node,
            "post_split": "scale_and_smote",
            "outputs": ["train_ready_ann", "test_ready_ann"]
        }
    }

    if model_name not in FEN_CONFIG:
        raise ValueError(f"Model {model_name} FEN config not found.")

    config = FEN_CONFIG[model_name]
    nodes_list = []

    # 1. Pre-split transformation
    # Note: identity_node only takes 1 input, apply_categorical_xgb takes 2
    if config["pre_split"] == identity_node:
        pre_split_inputs = "intermediate_credit_data"
    else:
        pre_split_inputs = ["intermediate_credit_data", "parameters"]

    nodes_list.append(node(
        func=config["pre_split"],
        inputs=pre_split_inputs,
        outputs=f"fen_input_{model_name}",
        name=f"pre_split_{model_name}_node",
        tags=[model_name, "feature_eng"]
    ))

    # 2. Split
    nodes_list.append(node(
        func=split_data,
        inputs=[f"fen_input_{model_name}", "parameters"],
        outputs=[f"raw_train_{model_name}", f"raw_test_{model_name}"],
        name=f"split_{model_name}_node",
        tags=[model_name, "feature_eng"]
    ))

    # 3. Post-split logic
    if config["post_split"] == "scale_and_smote":
        nodes_list.extend([
            node(
                func=scale_data_ann,
                inputs=[f"raw_train_{model_name}", f"raw_test_{model_name}"],
                outputs=[f"scaled_train_{model_name}", config["outputs"][1], f"fitted_scaler_{model_name}"],
                name=f"scale_{model_name}_node",
                tags=[model_name, "feature_eng"]
            ),
            node(
                func=smote_balance,
                inputs=[f"scaled_train_{model_name}", "parameters"],
                outputs=config["outputs"][0],
                name=f"smote_{model_name}_node",
                tags=[model_name, "feature_eng"]
            )
        ])
    else:
        # Default pass-through (No SMOTE for Boosting)
        nodes_list.extend([
            node(
                func=identity_node,
                inputs=f"raw_train_{model_name}",
                outputs=config["outputs"][0],
                name=f"pass_train_{model_name}_node",
                tags=[model_name, "feature_eng"]
            ),
            node(
                func=identity_node,
                inputs=f"raw_test_{model_name}",
                outputs=config["outputs"][1],
                name=f"pass_test_{model_name}_node",
                tags=[model_name, "feature_eng"]
            )
        ])
        
    return Pipeline(nodes_list)

def create_feature_eng_pipeline(**kwargs) -> Pipeline:
    # 1. Extraction phase (One separated initial extraction)
    extraction_node = node(
        func=preprocess_raw_data,
        inputs="raw_credit_data",
        outputs="intermediate_credit_data",
        name="preprocess_raw_data_node",
        tags=["feature_eng", "catboost", "xgboost", "ann"]
    )
    
    # 2. Model-specific FEN branches
    return Pipeline([extraction_node]) + \
           create_fen_model_pipeline("catboost") + \
           create_fen_model_pipeline("xgboost") + \
           create_fen_model_pipeline("ann")