"""
models/classification_models.py — Implementação e Treino de Classificadores
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report

from models.metrics import classification_metrics
from models.plots import plot_confusion_matrix, plot_all_confusion_matrices

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_classification_models() -> list:
    """Configura e retorna os classificadores para o projeto."""
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("SVC", SVC(kernel='rbf', probability=True, random_state=42))
    ]
    
    if HAS_XGBOOST:
        models.append(("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)))
    
    return models


def train_classification(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray,  y_test: np.ndarray,
                         class_names: list,   out_dir: Path) -> list:
    """Executa o pipeline completo de classificação: CV, Treino e Avaliação."""
    models     = get_classification_models()
    resultados = []
    all_preds  = {}
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    save_dir   = out_dir / "classificacao"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Classificação ──\n")

    for name, model in models:
        print(f"  A avaliar {name}...")

        # 1. Validação Cruzada (5-fold)
        scoring = ['accuracy', 'f1_macro']
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_acc = cv_results['test_accuracy'].mean()
        cv_f1  = cv_results['test_f1_macro'].mean()

        # 2. Treino final e Previsão
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_preds[name] = y_pred

        # 3. Cálculo de Métricas e Relatórios
        metrics = classification_metrics(y_test, y_pred, name)
        metrics["CV-Acc"] = round(cv_acc, 3)
        metrics["CV-F1"]  = round(cv_f1, 3)
        resultados.append(metrics)

        print(f"\n[Report: {name}]")
        print(classification_report(y_test, y_pred, target_names=[n.replace('\n', ' ') for n in class_names]))
        
        # 4. Geração de Matriz de Confusão
        plot_confusion_matrix(y_test, y_pred, class_names, name, out_dir)

        # 5. Persistência do Modelo
        fname = f"{name.lower().replace(' ', '_')}.pkl"
        with open(save_dir / fname, "wb") as f:
            pickle.dump(model, f)

    # Geração do gráfico comparativo total
    plot_all_confusion_matrices(y_test, all_preds, class_names, out_dir)

    return resultados
