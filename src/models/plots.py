"""
models/plots.py — Visualizações dos Resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

# Estilo global para os gráficos
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.size": 11,
    "axes.titlesize": 13,
})

COLORS = ["#3B8BD4", "#E8593C", "#EF9F27", "#534AB7"]


def plot_regression_comparison(resultados: list, out_dir: Path):
    """Gráfico de barras comparando as métricas entre modelos de regressão."""
    df    = pd.DataFrame(resultados).set_index("modelo")
    nomes = df.index.tolist()
    x     = np.arange(len(nomes))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Comparação de modelos — Regressão", fontweight="bold")

    for ax, metric, color in zip(axes, ["MAE", "RMSE", "MAPE"], COLORS):
        bars = ax.bar(x, df[metric], width=0.6, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(metric + (" (%)" if metric == "MAPE" else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, rotation=15, ha="right")
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = out_dir / "resultados/regressao"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "regressao_comparacao.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions_vs_real(model, X_test: np.ndarray, y_test: np.ndarray, name: str, out_dir: Path):
    """Gráfico de dispersão entre previsões e valores reais."""
    y_pred = np.clip(model.predict(X_test), 0, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="#3B8BD4", edgecolors="none")

    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Previsão perfeita")
    ax.set_xlabel("Contagem real")
    ax.set_ylabel("Contagem prevista")
    ax.set_title(f"Previsões vs Real — {name}")
    ax.legend()

    plt.tight_layout()
    save_path = out_dir / "resultados/regressao"
    save_path.mkdir(parents=True, exist_ok=True)
    fname = f"previsoes_vs_real_{name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, name: str, out_dir: Path):
    """Gera e guarda a matriz de confusão individual."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(f"Matriz de Confusão — {name}", fontweight="bold", pad=20)
    ax.grid(False)
    
    plt.tight_layout()
    save_path = out_dir / "resultados/classificacao"
    save_path.mkdir(parents=True, exist_ok=True)
    fname = f"cm_{name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_classification_comparison(resultados: list, out_dir: Path):
    """Gráfico de barras comparando Accuracy e F1-Score."""
    df    = pd.DataFrame(resultados).set_index("modelo")
    nomes = df.index.tolist()
    x     = np.arange(len(nomes))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Comparação de modelos — Classificação", fontweight="bold")

    for ax, metric, color in zip(axes, ["Accuracy", "F1-Score"], [COLORS[0], COLORS[2]]):
        bars = ax.bar(x, df[metric], width=0.5, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(nomes, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = out_dir / "resultados/classificacao"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "classificacao_comparacao.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_confusion_matrices(y_true: np.ndarray, all_preds: dict, class_names: list, out_dir: Path):
    """Gera um painel comparativo com as matrizes de confusão de todos os modelos."""
    n_models = len(all_preds)
    cols = 2
    rows = (n_models + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5))
    axes = axes.flatten()
    
    for i, (name, y_pred) in enumerate(all_preds.items()):
        ax = axes[i]
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Modelo: {name}", fontweight="bold", pad=10)
        ax.grid(False)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle("Matrizes de Confusão — Comparação Global", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = out_dir / "resultados/classificacao"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "cm_comparacao_total.png", dpi=150, bbox_inches="tight")
    plt.close()