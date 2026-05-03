from kedro.pipeline import Pipeline, node
from .nodes import preprocess_raw_data, no_fen_catboost, split_and_balance_data

def create_feature_eng_pipeline(**kwargs) -> Pipeline:
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
                func=no_fen_catboost,
                inputs=["intermediate_credit_data", "parameters"],
                outputs="catboost_ready_data",
                name="no_fen_catboost_node",
                tags=["catboost"],
            ),
            node(
                func=split_and_balance_data,
                inputs=["catboost_ready_data", "parameters"],
                outputs=[
                    "X_train_balanced_catboost", 
                    "X_test_catboost"
                ],
                name="split_and_balance_catboost_node",
                tags=["catboost"],
            ),
        ]
    )