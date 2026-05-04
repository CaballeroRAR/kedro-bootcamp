from kedro.pipeline import Pipeline, node
from .nodes import preprocess_raw_data, no_fen_boost, split_data, smote_balance

def feature_eng_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_raw_data,
                inputs="raw_credit_data",
                outputs="intermediate_credit_data",
                name="preprocess_raw_data_node",
            ),
            # CatBoost Branch
            node(
                func=no_fen_boost,
                inputs=["intermediate_credit_data", "parameters"],
                outputs="boost_ready_data", # No scalation
                name="no_fen_boost_node",
                tags=["no_fen_boost"],
            ),
            node(
                func=split_data,
                inputs=["boost_ready_data", "parameters"],
                outputs=[
                    "train_split_boost", 
                    "test_split_boost"
                ],
                name="split_data_boost_node",
                tags=["boost_split"],
            ),
            node(
                func=smote_balance,
                inputs=["train_split_boost", "parameters"],
                outputs="train_oversampled_boost",
                name="oversample_boost_node",
                tags=["boost_oversample"],
            ),
            
        ]
    )