"""
models/metrics.py — Funções de Cálculo de Métricas
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Median Absolute Percentage Error."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    """Calcula as métricas de regressão (MAE, RMSE, MAPE, MdAPE)."""
    return {
        "modelo": name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAPE":  round(mape(y_true, y_pred), 2),
        "MdAPE": round(mdape(y_true, y_pred), 2),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    """Calcula as métricas de classificação (Accuracy, F1-macro)."""
    return {
        "modelo":   name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "F1-Score": round(f1_score(y_true, y_pred, average='macro'), 3),
    }