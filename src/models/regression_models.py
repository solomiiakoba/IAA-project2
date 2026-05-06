"""
models/regression_models.py — Modelos de Regressão
Projeto: Crowd Counting com ML Clássico
PL3-G2 | IAA – Universidade de Aveiro

Modelos (complexidade crescente): Ridge, Random Forest, XGBoost / GB, SVR
Entregável 3: GridSearchCV para SVR e XGBoost.
"""

import pickle
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

from models.metrics import regression_metrics

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[AVISO] XGBoost não instalado. A usar GradientBoosting como alternativa.")


# Grids de hiperparâmetros para GridSearchCV
# Apenas SVR e XGBoost/GB são otimizados — os modelos lineares são
# suficientemente estáveis com os valores por omissão, e o Random Forest
# serve como baseline não-linear sem custo de tuning.

SVR_GRID = {
    "C":       [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5, 1.0],
    "kernel":  ["rbf"],           # rbf é o mais adequado para este espaço de features
}

XGBOOST_GRID = {
    "n_estimators":  [100, 300],
    "max_depth":     [3, 6],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],  # reduz overfitting ao treinar em subconjuntos
    "colsample_bytree": [0.8],
}

GB_GRID = {  # alternativa se XGBoost não estiver instalado
    "n_estimators":  [100, 300],
    "max_depth":     [3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}


def _grid_search(estimator, param_grid: dict, X_train: np.ndarray,
                 y_train: np.ndarray, cv: KFold, name: str):
    """
    Corre GridSearchCV com neg_mean_absolute_error como critério.
    Devolve o melhor estimador já ajustado em X_train completo.
    """
    print(f"    A correr GridSearchCV para {name}...")
    gs = GridSearchCV(
        estimator,
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,       # re-treina no conjunto completo com os melhores params
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"    Melhores hiperparâmetros ({name}): {gs.best_params_}")
    print(f"    CV-MAE (GridSearch)           : {-gs.best_score_:.2f}")
    return gs.best_estimator_, gs.best_params_, -gs.best_score_


def get_baseline_models() -> list:
    """
    Modelos treinados com hiperparâmetros fixos (sem grid search).
    Ridge e Random Forest são suficientemente estáveis para servir
    de baseline sem otimização exaustiva.
    """
    return [
        ("Ridge", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )),
    ]


def train_regression(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray,  y_test: np.ndarray,
                     out_dir: Path) -> list:
    """
    Treina todos os modelos de regressão e avalia no conjunto de teste.

    Fluxo por modelo:
      - Ridge / Random Forest : treino direto + 5-fold CV para estimar MAE
      - XGBoost / GB / SVR    : GridSearchCV (já inclui CV internamente),
                                 seguido de avaliação no teste com o melhor modelo

    Nenhuma informação do conjunto de teste é usada para selecionar
    hiperparâmetros (o GridSearchCV opera exclusivamente sobre X_train).
    """
    resultados = []
    cv         = KFold(n_splits=5, shuffle=True, random_state=42)
    save_dir   = out_dir / "regressao"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Regressão ──\n")

    # --- Modelos baseline (sem grid search) ---
    for name, model in get_baseline_models():
        print(f"  A treinar {name}...")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        cv_mae = -cv_scores.mean()

        model.fit(X_train, y_train)
        y_pred  = np.clip(model.predict(X_test), 0, None)
        metrics = regression_metrics(y_test, y_pred, name)
        metrics["CV-MAE (5-fold)"] = round(cv_mae, 2)
        metrics["best_params"]     = "—"
        resultados.append(metrics)

        print(f"    MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  "
              f"MAPE={metrics['MAPE']}%  MdAPE={metrics['MdAPE']}%  "
              f"CV-MAE={metrics['CV-MAE (5-fold)']}")

        _save_model(model, name, save_dir)

    # --- XGBoost ou GradientBoosting com GridSearchCV ---
    if HAS_XGBOOST:
        boosting_name  = "XGBoost"
        boosting_base  = XGBRegressor(random_state=42, verbosity=0)
        boosting_grid  = XGBOOST_GRID
    else:
        boosting_name  = "GradientBoosting"
        boosting_base  = GradientBoostingRegressor(random_state=42)
        boosting_grid  = GB_GRID

    print(f"\n  A otimizar {boosting_name} com GridSearchCV...")
    best_boosting, best_params_b, cv_mae_b = _grid_search(
        boosting_base, boosting_grid, X_train, y_train, cv, boosting_name
    )
    y_pred_b  = np.clip(best_boosting.predict(X_test), 0, None)
    metrics_b = regression_metrics(y_test, y_pred_b, boosting_name)
    metrics_b["CV-MAE (5-fold)"] = round(cv_mae_b, 2)
    metrics_b["best_params"]     = str(best_params_b)
    resultados.append(metrics_b)

    print(f"    MAE={metrics_b['MAE']}  RMSE={metrics_b['RMSE']}  "
          f"MAPE={metrics_b['MAPE']}%  MdAPE={metrics_b['MdAPE']}%  "
          f"CV-MAE={metrics_b['CV-MAE (5-fold)']}")

    _save_model(best_boosting, boosting_name, save_dir)

    # --- SVR com GridSearchCV ---
    print(f"\n  A otimizar SVR com GridSearchCV...")
    best_svr, best_params_s, cv_mae_s = _grid_search(
        SVR(), SVR_GRID, X_train, y_train, cv, "SVR"
    )
    y_pred_s  = np.clip(best_svr.predict(X_test), 0, None)
    metrics_s = regression_metrics(y_test, y_pred_s, "SVR")
    metrics_s["CV-MAE (5-fold)"] = round(cv_mae_s, 2)
    metrics_s["best_params"]     = str(best_params_s)
    resultados.append(metrics_s)

    print(f"    MAE={metrics_s['MAE']}  RMSE={metrics_s['RMSE']}  "
          f"MAPE={metrics_s['MAPE']}%  MdAPE={metrics_s['MdAPE']}%  "
          f"CV-MAE={metrics_s['CV-MAE (5-fold)']}")

    _save_model(best_svr, "SVR", save_dir)

    return resultados


def _save_model(model, name: str, save_dir: Path):
    fname = name.lower().replace(" ", "_") + ".pkl"
    with open(save_dir / fname, "wb") as f:
        pickle.dump(model, f)