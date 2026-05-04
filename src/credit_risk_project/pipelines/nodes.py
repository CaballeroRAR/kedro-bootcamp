import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Excel data using Pandas."""
    target_col = 'default payment next month'
    if target_col in df.columns:
        df = df.rename(columns={target_col: 'target'})
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    return df

def no_fen_boost(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Prepares data for CatBoost."""
    return df

def split_data(
    df: pd.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data for model training and testing."""
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=params["modeling"]["train_fraction"],
        random_state=params["modeling"]["random_seed"],
        stratify=y
    )
    
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    return train_df, test_df

def smote_balance(
    df_split: pd.DataFrame,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """Apply SMOTE with Pandas."""

    X = df_split.drop(columns=['target'])
    y = df_split['target']
    smote = SMOTE(random_state=params["modeling"]["random_seed"])

    X_res, y_res = smote.fit_resample(X, y)
    train_df_smote = pd.DataFrame(X_res, columns=X.columns)
    train_df_smote['target'] = y_res
    return train_df_smote   


def train_catboost(train_df: pd.DataFrame, params: Dict[str, Any]) -> CatBoostClassifier:
    """Train CatBoost Classifier."""
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    model_params = params["modeling"]["catboost"]
    cat_features = params["features"]["categorical"]
    
    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train, cat_features=cat_features)
    return model

def predict_probabilities(model: Any, test_df: pd.DataFrame) -> np.ndarray:
    """Generate probability predictions."""
    X_test = test_df.drop(columns=['target'])
    return model.predict_proba(X_test)[:, 1]

def evaluate_models(y_test: pd.DataFrame, y_prob: np.ndarray) -> Dict[str, float]:
    """Basic classification metrics."""
    y_true = y_test['target'].values
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob)
    }

def evaluate_calibration(y_test: pd.DataFrame, y_prob: np.ndarray) -> Dict[str, float]:
    """Yeh & Lien (2009) Calibration Check: Y = A + BX."""
    y_true = y_test['target'].values
    
    lr = LinearRegression()
    lr.fit(y_prob.reshape(-1, 1), y_true)
    
    return {
        "calibration_intercept_A": float(lr.intercept_),
        "calibration_slope_B": float(lr.coef_[0]),
        "r2_score": float(lr.score(y_prob.reshape(-1, 1), y_true))
    }