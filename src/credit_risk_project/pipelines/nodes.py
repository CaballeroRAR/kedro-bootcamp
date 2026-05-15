import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def identity_node(data: Any) -> Any:
    """Pass-through node for modular branching."""
    return data

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Excel data and standardize feature column names to X1-X23."""
    rename_map = {
        'LIMIT_BAL': 'X1',
        'SEX': 'X2',
        'EDUCATION': 'X3',
        'MARRIAGE': 'X4',
        'AGE': 'X5',
        'PAY_0': 'X6',
        'PAY_2': 'X7',
        'PAY_3': 'X8',
        'PAY_4': 'X9',
        'PAY_5': 'X10',
        'PAY_6': 'X11',
        'BILL_AMT1': 'X12',
        'BILL_AMT2': 'X13',
        'BILL_AMT3': 'X14',
        'BILL_AMT4': 'X15',
        'BILL_AMT5': 'X16',
        'BILL_AMT6': 'X17',
        'PAY_AMT1': 'X18',
        'PAY_AMT2': 'X19',
        'PAY_AMT3': 'X20',
        'PAY_AMT4': 'X21',
        'PAY_AMT5': 'X22',
        'PAY_AMT6': 'X23',
        'default payment next month': 'target'
    }
    
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    df = df.rename(columns=rename_map)
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

def scale_data_ann(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Applies Standard Scaling to numerical features for ANN and returns the fitted scaler."""
    X_train = train_df.drop(columns=['target'])
    X_test = test_df.drop(columns=['target'])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconstruct DataFrames
    train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=train_df.index)
    train_scaled['target'] = train_df['target']
    
    test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=test_df.index)
    test_scaled['target'] = test_df['target']
    
    return train_scaled, test_scaled, scaler

def apply_categorical_xgb(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Apply categorical type to features for XGBoost."""
    cat_features = params["features"]["categorical"]
    df[cat_features] = df[cat_features].astype("category")
    
    return df



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



# MODEL TRAIN NODES

def train_catboost(train_df: pd.DataFrame, params: Dict[str, Any]) -> CatBoostClassifier:
    """Train CatBoost Classifier."""
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    model_params = params["modeling"]["catboost"]
    cat_features = params["features"]["categorical"]
    
    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train, cat_features=cat_features)
    return model

def train_xgboost(train_df: pd.DataFrame, params: Dict[str, Any]) -> xgb.XGBClassifier:
    """Train XGBoost Classifier."""
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    model_params = params["modeling"]["xgboost"]
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def train_ann(train_df: pd.DataFrame, params: Dict[str, Any]) -> Any:
    """Placeholder for ANN training."""
    
    return None