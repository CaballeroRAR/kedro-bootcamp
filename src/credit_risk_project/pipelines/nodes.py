import polars as pl
from typing import Dict, Any, Tuple
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_raw_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean raw Excel data using Polars.
    """
    # Rename target column if standard name is found
    target_col = 'default payment next month'
    if target_col in df.columns:
        df = df.rename({target_col: 'target'})
    
    # Drop ID column if exists
    if 'ID' in df.columns:
        df = df.drop('ID')
        
    return df

def no_fen_catboost(df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """Prepares data for CatBoost with non feature engineering using Polars.
    Categorical features are kept as integers as they are already encoded.
    """
    return df

def split_and_balance_data(
    df: pl.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data and apply SMOTE. 
    Note: SMOTE requires conversion to NumPy internally.
    """
    # Prepare target and features
    X = df.drop('target').to_numpy()
    y = df['target'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=params["modeling"]["train_fraction"],
        random_state=params["modeling"]["random_seed"],
        stratify=y
    )
    
    # Apply SMOTE to balance classes in training set
    smote = SMOTE(random_state=params["modeling"]["random_seed"])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Reconstruct Polars DataFrames for the output
    feature_cols = df.drop('target').columns
    
    train_df = pl.from_numpy(X_train_res, schema=feature_cols)
    train_target = pl.from_numpy(y_train_res.reshape(-1, 1), schema=['target'])
    
    test_df = pl.from_numpy(X_test, schema=feature_cols)
    test_target = pl.from_numpy(y_test.reshape(-1, 1), schema=['target'])
    
    # Merge targets back or return as separate (keeping it consistent with previous signature)
    return train_df.with_columns(train_target), \
           test_df.with_columns(test_target)

def evaluate_models(
    y_test: pl.DataFrame, 
    y_prob: np.ndarray
) -> Dict[str, float]:
    """Calculate AUC, Brier Score, and Log Loss."""
    y_true = y_test.select('target').to_numpy().flatten()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob)
    }