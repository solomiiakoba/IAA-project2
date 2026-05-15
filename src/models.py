"""
models.py — Orquestrador do Treino (Regressão)
Projeto: Crowd Counting com ML Clássico
PL3-G2 | IAA – Universidade de Aveiro
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from models.regression_models import train_regression
from models.plots import plot_regression_comparison, plot_predictions_vs_real


def resolve_datasets(processed_root: Path, runs: list[str] | None) -> list[Path]:
    """Devolve lista de pastas de dados a processar."""
    if runs:
        paths = [processed_root / r for r in runs]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Pastas não encontradas: {missing}")
        return paths
    # sem argumento → todas as subpastas com X_train.npy
    return sorted(p for p in processed_root.iterdir() if (p / "X_train.npy").exists())


def run_on_dataset(data_dir: Path, out_dir: Path):
    """Treina e avalia modelos para um dataset processado."""
    run_name = data_dir.name
    run_out  = out_dir / run_name

    for sub in ["regressao", "resultados/regressao"]:
        (run_out / sub).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Dataset: {run_name}")
    print(f"{'='*50}\n")

    X_train     = np.load(data_dir / "X_train.npy")
    X_test      = np.load(data_dir / "X_test.npy")
    y_reg_train = np.load(data_dir / "y_reg_train.npy")
    y_reg_test  = np.load(data_dir / "y_reg_test.npy")

    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")

    res_reg = train_regression(X_train, y_reg_train, X_test, y_reg_test, run_out)

    pd.DataFrame(res_reg).to_csv(run_out / "resultados/regressao/regressao_metricas.csv", index=False)

    plot_regression_comparison(res_reg, run_out)

    best_name = min(res_reg, key=lambda x: x["MAPE"])["modelo"]
    best_path = run_out / "regressao" / f"{best_name.lower().replace(' ', '_')}.pkl"
    with open(best_path, "rb") as f:
        best_model = pickle.load(f)
    plot_predictions_vs_real(best_model, X_test, y_reg_test, best_name, run_out)

    print(f"\n  Melhor modelo: {best_name}")
    print(pd.DataFrame(res_reg).to_string(index=False))
    return run_name, res_reg


def main():
    parser = argparse.ArgumentParser(description="Treino dos modelos de regressão")
    parser.add_argument("--processed_dir", type=str, default="./processed",
                        help="Pasta raiz com os datasets processados")
    parser.add_argument("--out_dir", type=str, default="./models_output",
                        help="Pasta onde guardar modelos e resultados")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Subpastas a usar (ex: exp1 exp2). Omitir = todas.")
    args = parser.parse_args()

    processed_root = Path(args.processed_dir)
    out_dir        = Path(args.out_dir)

    datasets = resolve_datasets(processed_root, args.runs)
    if not datasets:
        print("Nenhum dataset encontrado em", processed_root)
        return

    print(f"\nDatasets a processar: {[d.name for d in datasets]}")

    all_results = {}
    for data_dir in datasets:
        run_name, res = run_on_dataset(data_dir, out_dir)
        all_results[run_name] = res

    # Sumário comparativo entre runs (se mais de um)
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("  SUMÁRIO COMPARATIVO")
        print(f"{'='*50}")
        rows = []
        for run_name, res in all_results.items():
            best = min(res, key=lambda x: x["MAPE"])
            rows.append({"run": run_name, **{k: best[k] for k in ["modelo", "MAE", "RMSE", "MAPE"]}})
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nResultados guardados em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
