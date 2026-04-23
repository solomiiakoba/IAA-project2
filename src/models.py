"""
models.py — Orquestrador do Treino
Projeto: Crowd Counting com ML Clássico
PL3-G2 | IAA – Universidade de Aveiro

Uso:
    python models.py --data_dir ./processed

Gera:
    models_output/
        regressao/          ← modelos .pkl
        classificacao/      ← modelos .pkl
        resultados/         ← gráficos + CSV com métricas
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from models.regression_models import train_regression
from models.classification_models import train_classification
from models.plots import (
    plot_regression_comparison,
    plot_predictions_vs_real,
    plot_classification_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Treino dos modelos ML")
    parser.add_argument("--data_dir", type=str, default="./processed",
                        help="Pasta com os .npy do preprocessing")
    parser.add_argument("--out_dir", type=str, default="./models_output",
                        help="Pasta onde guardar modelos e resultados")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    for sub in ["regressao", "classificacao", "resultados"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print("\n=== Treino dos Modelos ===\n")

    # Carregar dados
    print("A carregar dados processados...")
    X_train     = np.load(data_dir / "X_train.npy")
    X_test      = np.load(data_dir / "X_test.npy")
    y_reg_train = np.load(data_dir / "y_reg_train.npy")
    y_reg_test  = np.load(data_dir / "y_reg_test.npy")

    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")

    # 1. Criar as Classes (Discretização da Variável Alvo)
    print("A discretizar a variável alvo para classificação...")
    
    def discretize(y):
        y_class = np.zeros_like(y, dtype=int)
        y_class[y <= 102] = 0                           # Baixa densidade
        y_class[(y > 102) & (y <= 248)] = 1             # Moderada
        y_class[y > 248] = 2                            # Sobrelotação
        return y_class

    y_train_class = discretize(y_reg_train)
    y_test_class  = discretize(y_reg_test)

    # Nomes das classes
    class_names = [
        "Baixa\n(≤ 102)",
        "Moderada\n(103-248)",
        "Sobrelotação\n(> 248)"
    ]

    print(f"  Distribuição classes (train): {np.bincount(y_train_class)}")

    # Treino 
    res_reg   = train_regression(X_train, y_reg_train, X_test, y_reg_test, out_dir)
    res_class = train_classification(X_train, y_train_class, X_test, y_test_class, class_names, out_dir)

    # Guardar métricas
    pd.DataFrame(res_reg).to_csv(out_dir / "resultados" / "regressao_metricas.csv",  index=False)
    pd.DataFrame(res_class).to_csv(out_dir / "resultados" / "classificacao_metricas.csv", index=False)

    # Gráficos 
    print("\nA gerar gráficos...")
    plot_regression_comparison(res_reg, out_dir)
    plot_classification_comparison(res_class, out_dir)

    # Scatter do melhor modelo de regressão (menor MAPE)
    best_name = min(res_reg, key=lambda x: x["MAPE"])["modelo"]
    print(f"\n  Melhor modelo regressão (MAPE): {best_name}")
    best_path = out_dir / "regressao" / f"{best_name.lower().replace(' ', '_')}.pkl"
    with open(best_path, "rb") as f:
        best_model = pickle.load(f)
    plot_predictions_vs_real(best_model, X_test, y_reg_test, best_name, out_dir)

    # Resumo
    print("\n=== Resultados Finais ===\n")
    print("── Regressão ──")
    print(pd.DataFrame(res_reg).to_string(index=False))
    
    print("\n── Classificação ──")
    print(pd.DataFrame(res_class).to_string(index=False))
    
    print(f"\nFicheiros em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()