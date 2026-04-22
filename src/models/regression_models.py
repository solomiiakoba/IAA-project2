"""
models/regression_models.py - Modelos de Regressão
Modelos (complexidade crescente): Ridge, Random Forest, XGBoost / GB, SVR 
"""

import pickle
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold

from models.metrics import regression_metrics

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[AVISO] XGBoost não instalado. A usar GradientBoosting como alternativa.")


def get_models() -> list:
    """
    Define os modelos de regressão e os seus hiperparâmetros iniciais.
    A seleção final de hiperparâmetros deve ser feita com GridSearchCV
    !!! no deliverable 3
    """
    models = [
        ("Ridge", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )),
    ]

    if HAS_XGBOOST:
        models.append(("XGBoost", XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )))
    else:
        models.append(("GradientBoosting", GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )))

    models.append(("SVR", SVR(kernel="rbf", C=10, epsilon=0.1)))

    return models


def train_regression(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray,  y_test: np.ndarray,
                     out_dir: Path) -> list:
    """
    Treina todos os modelos de regressão, avalia no conjunto de teste
    e guarda cada modelo.
    Utiliza 5-fold cross-validation no treino para estimar o MAE
    sem tocar no conjunto de teste.
    """
    models     = get_models()
    resultados = []
    cv         = KFold(n_splits=5, shuffle=True, random_state=42)
    save_dir   = out_dir / "regressao"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Regressão ──\n")

    for name, model in models:
        print(f"  A treinar {name}...")

        # Cross-validation no treino
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        cv_mae = -cv_scores.mean()

        # Treino final
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)  # sem contagens negativas

        metrics = regression_metrics(y_test, y_pred, name)
        metrics["CV-MAE (5-fold)"] = round(cv_mae, 2)
        resultados.append(metrics)

        print(f"    MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  "
              f"MAPE={metrics['MAPE']}%  MdAPE={metrics['MdAPE']}%  "
              f"CV-MAE={metrics['CV-MAE (5-fold)']}")

        # Guardar modelo
        fname = name.lower().replace(" ", "_") + ".pkl"
        with open(save_dir / fname, "wb") as f:
            pickle.dump(model, f)

    return resultados