"""
models/metrics.py — Métricas de Avaliação
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# Regressão

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Ignora amostras com y_true == 0 para evitar divisão por zero.
    Preferido ao RMSE quando a distribuição tem cauda longa.
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error (para outliers)."""
    mask = y_true != 0
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    """Calcula MAE, RMSE, MAPE e MdAPE para um modelo de regressão."""
    return {
        "modelo": name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAPE":  round(mape(y_true, y_pred), 2),
        "MdAPE": round(mdape(y_true, y_pred), 2),
    }

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)