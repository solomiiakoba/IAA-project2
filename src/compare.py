"""
compare.py — Comparação de resultados entre diferentes configurações de preprocessing
Lê os CSVs de métricas gerados pelo models.py e produz tabelas e gráficos comparativos.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.size": 10,
    "axes.titlesize": 11,
})

METRICS = ["MAE", "RMSE", "MAPE", "MdAPE", "CV-MAE (5-fold)"]


def load_all_results(models_dir: Path, runs: list[str] | None) -> pd.DataFrame:
    """Carrega todos os CSVs de métricas e concatena num único DataFrame."""
    if runs:
        subdirs = [models_dir / r for r in runs]
    else:
        subdirs = sorted(models_dir.iterdir())

    rows = []
    for subdir in subdirs:
        csv = subdir / "resultados" / "regressao" / "regressao_metricas.csv"
        if not csv.exists():
            print(f"  [AVISO] Sem resultados em: {subdir.name}")
            continue
        df = pd.read_csv(csv)
        df.insert(0, "run", subdir.name)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"Nenhum resultado encontrado em {models_dir}")

    return pd.concat(rows, ignore_index=True)


def plot_comparison(df: pd.DataFrame, out_path: Path):
    """Gráfico de barras agrupadas: cada métrica, barras por run+modelo."""
    runs   = df["run"].unique()
    models = df["modelo"].unique()
    n_runs = len(runs)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_runs))
    run_color = dict(zip(runs, colors))

    metrics_to_plot = [m for m in METRICS if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    fig.suptitle("Comparação entre configurações de preprocessing", fontweight="bold", y=1.02)

    for ax, metric in zip(axes, metrics_to_plot):
        x      = np.arange(len(models))
        width  = 0.8 / n_runs
        for i, run in enumerate(runs):
            vals = [
                df[(df["run"] == run) & (df["modelo"] == m)][metric].values
                for m in models
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - n_runs / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                        color=run_color[run], alpha=0.85, label=run)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")

    axes[-1].legend(title="run", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out_path}")


def plot_best_per_run(df: pd.DataFrame, out_path: Path):
    """Gráfico com o melhor modelo (menor MAPE) de cada run."""
    best_rows = df.loc[df.groupby("run")["MAPE"].idxmin()].copy()

    metrics_to_plot = [m for m in METRICS if m in df.columns]
    runs   = best_rows["run"].tolist()
    x      = np.arange(len(runs))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(runs)))

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    fig.suptitle("Melhor modelo por configuração (menor MAPE)", fontweight="bold", y=1.02)

    for ax, metric in zip(axes, metrics_to_plot):
        vals = best_rows[metric].tolist()
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r}\n({m})" for r, m in zip(best_rows["run"], best_rows["modelo"])],
            rotation=20, ha="right", fontsize=8
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparação de resultados entre runs")
    parser.add_argument("--models_dir", type=str, default="./models_output",
                        help="Pasta raiz com os resultados dos modelos")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Runs a comparar (omitir = todos)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Pasta de output dos gráficos (default: --models_dir/comparacao)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    out_dir    = Path(args.out_dir) if args.out_dir else models_dir / "comparacao"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Comparação de Resultados ===\n")

    df = load_all_results(models_dir, args.runs)

    # Tabela completa
    print(df.to_string(index=False))
    df.to_csv(out_dir / "comparacao_completa.csv", index=False)
    print(f"\n  CSV guardado: {out_dir / 'comparacao_completa.csv'}")

    # Melhor modelo por run
    best = df.loc[df.groupby("run")["MAPE"].idxmin(), ["run", "modelo"] + [m for m in METRICS if m in df.columns]]
    print("\n── Melhor modelo por run (menor MAPE) ──")
    print(best.to_string(index=False))
    best.to_csv(out_dir / "comparacao_melhores.csv", index=False)

    # Gráficos
    print("\nA gerar gráficos...")
    plot_comparison(df, out_dir / "comparacao_todos_modelos.png")
    plot_best_per_run(df, out_dir / "comparacao_melhores.png")

    print(f"\nResultados guardados em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
