"""
models/classification_models.py - Modelos de Classificação
Projeto: Crowd Counting com ML Clássico
Modelos: Logistic Regression, Random Forest, XGBoost, SVC
"""

import pickle
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold

from models.metrics import classification_metrics
from models.plots import plot_confusion_matrix
from sklearn.metrics import classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[AVISO] XGBoost não instalado para classificação.")


def get_classification_models() -> list:
    """
    Define os modelos de classificação e os seus hiperparâmetros iniciais.
    """
    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, 
            random_state=42,
            multi_class='multinomial'
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )),
        ("SVC", SVC(
            kernel="rbf", 
            C=1.0,
            probability=True, 
            random_state=42
        ))
    ]

    if HAS_XGBOOST:
        models.append(("XGBoost", XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            objective='multi:softprob'
        )))
    
    return models


def train_classification(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray,  y_test: np.ndarray,
                         class_names: list, out_dir: Path) -> list:
    """
    Treina os modelos de classificação, avalia e guarda os resultados.
    """
    models     = get_classification_models()
    resultados = []
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    save_dir   = out_dir / "classificacao"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Classificação ──\n")

    for name, model in models:
        print(f"  A treinar {name}...")

        # Validação Cruzada (5-fold) no treino
        scoring = ['accuracy', 'f1_macro']
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_acc = cv_results['test_accuracy'].mean()
        cv_f1  = cv_results['test_f1_macro'].mean()

        # Treino final
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = classification_metrics(y_test, y_pred, name)
        metrics["CV-Acc"] = round(cv_acc, 3)
        metrics["CV-F1"]  = round(cv_f1, 3)
        resultados.append(metrics)

        print(f"    Acc={metrics['Accuracy']}  F1={metrics['F1-Score']}  "
              f"CV-Acc={metrics['CV-Acc']}  CV-F1={metrics['CV-F1']}")

        # Report e Matriz de Confusão
        print(f"\nReport para {name}:")
        print(classification_report(y_test, y_pred, target_names=[n.replace('\n', ' ') for n in class_names]))
        plot_confusion_matrix(y_test, y_pred, class_names, name, out_dir)

        # Guardar modelo
        fname = name.lower().replace(" ", "_") + ".pkl"
        with open(save_dir / fname, "wb") as f:
            pickle.dump(model, f)

    return resultados
