from kedro.pipeline import Pipeline, node
from .nodes import train_catboost, predict_probabilities, evaluate_models, evaluate_calibration

def create_model_specific_pipeline(model_name: str) -> Pipeline:
    """Factory to create a training pipeline for a specific model."""
    return Pipeline(
        [
            node(
                func=train_catboost if model_name == "catboost" else None, # Placeholder for others
                inputs=[f"X_train_balanced_{model_name}", "parameters"],
                outputs=f"{model_name}_model",
                name=f"train_{model_name}_node",
                tags=[model_name],
            ),
            node(
                func=predict_probabilities,
                inputs=[f"{model_name}_model", f"X_test_{model_name}"],
                outputs=f"y_prob_{model_name}",
                name=f"predict_{model_name}_node",
                tags=[model_name],
            ),
            node(
                func=evaluate_models,
                inputs=[f"y_test_{model_name}", f"y_prob_{model_name}"],
                outputs=f"{model_name}_metrics",
                name=f"evaluate_{model_name}_node",
                tags=[model_name],
            ),
            node(
                func=evaluate_calibration,
                inputs=[f"y_test_{model_name}", f"y_prob_{model_name}"],
                outputs=f"{model_name}_calibration",
                name=f"calibrate_{model_name}_node",
                tags=[model_name],
            ),
        ]
    )

def create_training_pipeline(**kwargs) -> Pipeline:
    """Aggregates all model pipelines into a single training entry point."""
    catboost_pipeline = create_model_specific_pipeline("catboost")
    # xg_pipeline = create_model_specific_pipeline("xgboost")
    # ann_pipeline = create_model_specific_pipeline("ann")
    
    return catboost_pipeline # + xg_pipeline + ann_pipeline