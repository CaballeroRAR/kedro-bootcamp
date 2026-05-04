from kedro.pipeline import Pipeline, node
from .nodes import train_catboost, train_xgboost, train_ann

def create_model_specific_pipeline(model_name: str) -> Pipeline:
    """Factory to create a training-only pipeline for a specific model."""
    
    # Model Configuration Dispatcher
    MODEL_CONFIG = {
        "catboost": {
            "train_func": train_catboost,
            "train_ds": "train_oversampled_boost",
        },
        "xgboost": {
            "train_func": train_xgboost,
            "train_ds": "train_categorical_xgboost",
        },
        "ann": {
            "train_func": train_ann,
            "train_ds": "train_scaled_ann", # Placeholder
        },
    }

    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Model {model_name} not supported by training factory.")

    config = MODEL_CONFIG[model_name]

    return Pipeline(
        [
            node(
                func=config["train_func"],
                inputs=[config["train_ds"], "parameters"],
                outputs=f"{model_name}_model",
                name=f"train_{model_name}_node",
                tags=[model_name, "training"],
            ),
        ]
    )

def create_training_pipeline(**kwargs) -> Pipeline:
    """Aggregates all training-only pipelines."""
    return (
        create_model_specific_pipeline("catboost") +
        create_model_specific_pipeline("xgboost") +
        create_model_specific_pipeline("ann")
    )