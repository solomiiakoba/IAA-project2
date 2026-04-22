"""
eda.py produz análise exploratória do Dataset ShanghaiTech
Uso:
    python eda.py --data_root ./data/ShanghaiTech

Gera:
    eda_output/
        01_distribuicao_contagens.png
        02_comparacao_partA_partB.png
        03_cauda_longa.png
        04_limiares_classificacao.png
        05_exemplos_imagens.png
        eda_report.txt
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path

# ──────────────────────────────────────────────
# Configuração de estilo dos gráficos
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COLORS = {
    "partA": "#E8593C",
    "partB": "#3B8BD4",
    "baixa": "#3B8BD4",
    "moderada": "#EF9F27",
    "sobrelotacao": "#E24B4A",
}


# ──────────────────────────────────────────────
# Carregamento dos dados
# ──────────────────────────────────────────────

def load_counts(data_root: str) -> dict:
    """
    Percorre Part A e Part B (train + test) e extrai contagem de pessoas por imagem, nome do ficheiro, resolução da imagem
    """
    data_root = Path(data_root)
    results = {}

    for part in ["part_A", "part_B"]:
        part_path = data_root / part
        if not part_path.exists():
            print(f"[AVISO] Pasta não encontrada: {part_path}")
            continue

        for split in ["train_data", "test_data"]:
            split_path = part_path / split
            gt_path = split_path / "ground-truth"
            img_path = split_path / "images"

            if not gt_path.exists():
                print(f"[AVISO] Ground truth não encontrada: {gt_path}")
                continue

            counts = []
            resolutions = []
            filenames = []

            gt_files = sorted(gt_path.glob("GT_*.mat"))
            for gt_file in gt_files:
                try:
                    mat = sio.loadmat(str(gt_file))
                    locations = mat["image_info"][0][0][0][0][0]
                    n_pessoas = len(locations)
                    counts.append(n_pessoas)
                    filenames.append(gt_file.name)

                    # Resolução da imagem correspondente
                    img_name = gt_file.name.replace("GT_", "").replace(".mat", ".jpg")
                    img_file = img_path / img_name
                    if img_file.exists():
                        with Image.open(img_file) as im:
                            resolutions.append(im.size)  # (width, height)
                    else:
                        resolutions.append(None)

                except Exception as e:
                    print(f"[ERRO] {gt_file.name}: {e}")

            key = f"{part}_{split}"
            results[key] = {
                "counts": np.array(counts),
                "filenames": filenames,
                "resolutions": resolutions,
                "part": part,
                "split": split,
            }
            print(f"  Carregado {key}: {len(counts)} imagens")

    return results


def merge_part(data: dict, part: str) -> np.ndarray:
    """Junta train + test de uma parte."""
    arrays = []
    for key, val in data.items():
        if val["part"] == part:
            arrays.append(val["counts"])
    return np.concatenate(arrays) if arrays else np.array([])


# ──────────────────────────────────────────────
# Gráficos
# ──────────────────────────────────────────────

def plot_distribuicao(counts_A, counts_B, out_dir):
    """Histogramas sobrepostos de Part A e Part B."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuição do número de pessoas por imagem", fontweight="bold")

    for ax, counts, part, color in zip(
        axes,
        [counts_A, counts_B],
        ["Part A (multidões densas)", "Part B (multidões esparsas)"],
        [COLORS["partA"], COLORS["partB"]],
    ):
        ax.hist(counts, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(np.mean(counts), color="black", linestyle="--",
                   linewidth=1.5, label=f"Média: {np.mean(counts):.0f}")
        ax.axvline(np.median(counts), color="gray", linestyle=":",
                   linewidth=1.5, label=f"Mediana: {np.median(counts):.0f}")
        ax.set_title(part)
        ax.set_xlabel("Nº de pessoas")
        ax.set_ylabel("Nº de imagens")
        ax.legend()

    plt.tight_layout()
    path = out_dir / "01_distribuicao_contagens.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


def plot_comparacao(counts_A, counts_B, out_dir):
    """Boxplot comparativo + estatísticas lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparação Part A vs Part B", fontweight="bold")

    # Boxplot
    bp = axes[0].boxplot(
        [counts_A, counts_B],
        labels=["Part A", "Part B"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor(COLORS["partA"] + "88")
    bp["boxes"][1].set_facecolor(COLORS["partB"] + "88")
    axes[0].set_ylabel("Nº de pessoas")
    axes[0].set_title("Boxplot")

    # Estatísticas em tabela
    stats = {
        "Mínimo": [counts_A.min(), counts_B.min()],
        "Percentil 25": [np.percentile(counts_A, 25), np.percentile(counts_B, 25)],
        "Mediana": [np.median(counts_A), np.median(counts_B)],
        "Média": [counts_A.mean(), counts_B.mean()],
        "Percentil 75": [np.percentile(counts_A, 75), np.percentile(counts_B, 75)],
        "Máximo": [counts_A.max(), counts_B.max()],
        "Desvio padrão": [counts_A.std(), counts_B.std()],
    }
    col_labels = ["Part A", "Part B"]
    row_labels = list(stats.keys())
    cell_text = [[f"{stats[k][0]:.1f}", f"{stats[k][1]:.1f}"] for k in row_labels]

    axes[1].axis("off")
    table = axes[1].table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    axes[1].set_title("Estatísticas descritivas")

    plt.tight_layout()
    path = out_dir / "02_comparacao_partA_partB.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


def plot_cauda_longa(counts_A, counts_B, out_dir):
    """
    Demonstra a cauda longa: histograma em escala log + CDF cumulativa.
    Justifica o uso de MAPE em vez de RMSE.
    """
    all_counts = np.concatenate([counts_A, counts_B])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cauda longa da distribuição — justificação para usar MAPE", fontweight="bold")

    # Histograma escala log
    axes[0].hist(all_counts, bins=40, color="#534AB7", alpha=0.8, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Nº de pessoas")
    axes[0].set_ylabel("Nº de imagens (escala log)")
    axes[0].set_title("Histograma em escala logarítmica")

    # Anotação explicativa
    axes[0].annotate(
        "Cauda longa:\npoucas imagens com\nmuitas pessoas",
        xy=(all_counts.max() * 0.6, 1.5),
        fontsize=9,
        color="#E24B4A",
        ha="center",
    )

    # CDF
    sorted_counts = np.sort(all_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1].plot(sorted_counts, cdf, color="#534AB7", linewidth=2)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% das imagens")
    axes[1].axhline(0.9, color="#E8593C", linestyle="--", linewidth=1, label="90% das imagens")
    axes[1].set_xlabel("Nº de pessoas")
    axes[1].set_ylabel("Proporção acumulada")
    axes[1].set_title("Função de distribuição acumulada (CDF)")
    axes[1].legend()

    # Anotações nos percentis
    p50 = np.percentile(all_counts, 50)
    p90 = np.percentile(all_counts, 90)
    axes[1].axvline(p50, color="gray", linestyle=":", linewidth=1)
    axes[1].axvline(p90, color="#E8593C", linestyle=":", linewidth=1)
    axes[1].text(p50, 0.05, f" p50={p50:.0f}", fontsize=9, color="gray")
    axes[1].text(p90, 0.05, f" p90={p90:.0f}", fontsize=9, color="#E8593C")

    plt.tight_layout()
    path = out_dir / "03_cauda_longa.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


def plot_limiares_classificacao(counts_A, counts_B, out_dir):
    """
    Define limiares para classificação por percentis e visualiza as 3 classes.
    Retorna os limiares para usar no preprocessing.
    """
    all_counts = np.concatenate([counts_A, counts_B])

    # Limiares baseados em percentis (classes equilibradas)
    p33 = np.percentile(all_counts, 33)
    p66 = np.percentile(all_counts, 66)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Definição de limiares para classificação por densidade", fontweight="bold")

    # Histograma com zonas coloridas
    bins = np.linspace(all_counts.min(), all_counts.max(), 40)
    axes[0].hist(all_counts[all_counts <= p33], bins=bins,
                 color=COLORS["baixa"], alpha=0.8, label=f"Baixa (≤{p33:.0f})")
    axes[0].hist(all_counts[(all_counts > p33) & (all_counts <= p66)], bins=bins,
                 color=COLORS["moderada"], alpha=0.8, label=f"Moderada ({p33:.0f}–{p66:.0f})")
    axes[0].hist(all_counts[all_counts > p66], bins=bins,
                 color=COLORS["sobrelotacao"], alpha=0.8, label=f"Sobrelotação (>{p66:.0f})")
    axes[0].axvline(p33, color="black", linestyle="--", linewidth=1.5)
    axes[0].axvline(p66, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Nº de pessoas")
    axes[0].set_ylabel("Nº de imagens")
    axes[0].set_title("Distribuição com classes de densidade")
    axes[0].legend()

    # Proporção de imagens por classe
    classes = ["Baixa", "Moderada", "Sobrelotação"]
    proporcoes = [
        np.sum(all_counts <= p33),
        np.sum((all_counts > p33) & (all_counts <= p66)),
        np.sum(all_counts > p66),
    ]
    cores = [COLORS["baixa"], COLORS["moderada"], COLORS["sobrelotacao"]]
    bars = axes[1].bar(classes, proporcoes, color=cores, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Nº de imagens")
    axes[1].set_title("Equilíbrio das classes")
    for bar, n in zip(bars, proporcoes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(n), ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    path = out_dir / "04_limiares_classificacao.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

    return p33, p66


def plot_exemplos(data: dict, out_dir):
    """
    Mostra 6 exemplos de imagens: 2 por classe de densidade.
    Ajuda a perceber visualmente o que cada classe representa.
    """
    # Junta todos os dados
    all_entries = []
    for key, val in data.items():
        part_path = Path(val.get("base_path", ""))
        for fname, count in zip(val["filenames"], val["counts"]):
            all_entries.append((fname, count, val["part"], val["split"], val.get("base_path", "")))

    if not all_entries:
        print("  [AVISO] Sem caminhos de imagens para exemplos visuais.")
        return

    # Filtra por classe
    all_counts = np.array([e[1] for e in all_entries])
    p33 = np.percentile(all_counts, 33)
    p66 = np.percentile(all_counts, 66)

    baixa = [e for e in all_entries if e[1] <= p33]
    moderada = [e for e in all_entries if p33 < e[1] <= p66]
    alta = [e for e in all_entries if e[1] > p66]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Exemplos visuais por classe de densidade", fontweight="bold")

    grupos = [
        (baixa[:2], "Baixa densidade", COLORS["baixa"]),
        (moderada[:2], "Densidade moderada", COLORS["moderada"]),
        (alta[:2], "Sobrelotação", COLORS["sobrelotacao"]),
    ]

    for col, (grupo, titulo, cor) in enumerate(grupos):
        axes[0, col].set_title(titulo, color=cor, fontweight="bold")
        for row, entry in enumerate(grupo[:2]):
            fname, count, part, split, base_path = entry
            img_name = fname.replace("GT_", "").replace(".mat", ".jpg")
            img_path = Path(base_path) / "images" / img_name

            ax = axes[row, col]
            if img_path.exists():
                img = np.array(Image.open(img_path).convert("RGB"))
                ax.imshow(img)
                ax.set_title(f"{count} pessoas", fontsize=10)
            else:
                ax.text(0.5, 0.5, f"{count} pessoas\n(imagem não encontrada)",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#eee")
            ax.axis("off")

    plt.tight_layout()
    path = out_dir / "05_exemplos_imagens.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")


# ──────────────────────────────────────────────
# Relatório 
# ──────────────────────────────────────────────

def gerar_relatorio(data: dict, p33: float, p66: float, out_dir: Path):
    """Gera um ficheiro .txt com as principais estatísticas para incluir no relatório."""
    lines = []
    lines.append("=" * 60)
    lines.append("RELATÓRIO EDA — ShanghaiTech Crowd Counting")
    lines.append("=" * 60)
    lines.append("")

    all_counts_A, all_counts_B = [], []

    for key, val in data.items():
        counts = val["counts"]
        lines.append(f"── {key} ──")
        lines.append(f"  Nº imagens   : {len(counts)}")
        lines.append(f"  Mínimo       : {counts.min()}")
        lines.append(f"  Máximo       : {counts.max()}")
        lines.append(f"  Média        : {counts.mean():.1f}")
        lines.append(f"  Mediana      : {np.median(counts):.1f}")
        lines.append(f"  Desvio padrão: {counts.std():.1f}")
        lines.append(f"  Percentil 25 : {np.percentile(counts, 25):.1f}")
        lines.append(f"  Percentil 75 : {np.percentile(counts, 75):.1f}")
        lines.append("")

        if val["part"] == "part_A":
            all_counts_A.extend(counts.tolist())
        else:
            all_counts_B.extend(counts.tolist())

    all_counts = np.array(all_counts_A + all_counts_B)

    lines.append("── Dataset completo (A + B) ──")
    lines.append(f"  Total de imagens: {len(all_counts)}")
    lines.append(f"  Média global    : {all_counts.mean():.1f}")
    lines.append(f"  Mediana global  : {np.median(all_counts):.1f}")
    lines.append("")
    lines.append("── Limiares de classificação (percentis 33/66) ──")
    lines.append(f"  Baixa densidade : ≤ {p33:.0f} pessoas")
    lines.append(f"  Moderada        : {p33:.0f} – {p66:.0f} pessoas")
    lines.append(f"  Sobrelotação    : > {p66:.0f} pessoas")
    lines.append("")
    n_baixa = np.sum(all_counts <= p33)
    n_mod = np.sum((all_counts > p33) & (all_counts <= p66))
    n_alta = np.sum(all_counts > p66)
    lines.append(f"  Nº imagens — Baixa      : {n_baixa} ({100*n_baixa/len(all_counts):.1f}%)")
    lines.append(f"  Nº imagens — Moderada   : {n_mod} ({100*n_mod/len(all_counts):.1f}%)")
    lines.append(f"  Nº imagens — Sobrelotação: {n_alta} ({100*n_alta/len(all_counts):.1f}%)")
    lines.append("")
    lines.append("── Justificação para uso de MAPE ──")
    lines.append("  A distribuição apresenta cauda longa acentuada.")
    lines.append("  O RMSE penaliza desproporcionalmente erros em imagens")
    lines.append("  com muitas pessoas (outliers de contagem elevada).")
    lines.append("  O MAPE normaliza o erro pelo valor real, sendo mais")
    lines.append("  adequado para comparações entre imagens com densidades")
    lines.append("  muito diferentes (5 pessoas vs 3000 pessoas).")
    lines.append("")
    lines.append("  Métricas recomendadas:")
    lines.append("    Regressão   → MAE, RMSE, MAPE, MdAPE")
    lines.append("    Classificação → Accuracy, F1-macro, Matriz de confusão")

    report_path = out_dir / "eda_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Guardado: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="EDA do ShanghaiTech dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/ShanghaiTech",
        help="Caminho para a pasta raiz do ShanghaiTech",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./eda_output",
        help="Pasta onde guardar os gráficos e relatório",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EDA — ShanghaiTech Crowd Counting ===\n")
    print("A carregar dados...")

    data = load_counts(args.data_root)

    if not data:
        print("[ERRO] Nenhum dado carregado. Verifica o caminho --data_root.")
        return

    # Guarda base_path em cada entrada para aceder às imagens
    for key, val in data.items():
        part = val["part"]
        split = val["split"]
        val["base_path"] = str(Path(args.data_root) / part / split)

    counts_A = merge_part(data, "part_A")
    counts_B = merge_part(data, "part_B")

    print(f"\nPart A: {len(counts_A)} imagens | Part B: {len(counts_B)} imagens")
    print("\nA gerar gráficos...\n")

    if len(counts_A) > 0 and len(counts_B) > 0:
        plot_distribuicao(counts_A, counts_B, out_dir)
        plot_comparacao(counts_A, counts_B, out_dir)
        plot_cauda_longa(counts_A, counts_B, out_dir)
        p33, p66 = plot_limiares_classificacao(counts_A, counts_B, out_dir)
    elif len(counts_A) > 0:
        all_c = counts_A
        p33, p66 = np.percentile(all_c, 33), np.percentile(all_c, 66)
        plot_limiares_classificacao(all_c, np.array([]), out_dir)
    else:
        all_c = counts_B
        p33, p66 = np.percentile(all_c, 33), np.percentile(all_c, 66)
        plot_limiares_classificacao(np.array([]), all_c, out_dir)

    plot_exemplos(data, out_dir)
    gerar_relatorio(data, p33, p66, out_dir)

    print("\n=== EDA concluída ===")
    print(f"Ficheiros guardados em: {out_dir.resolve()}")
    print(f"\nLimiares para classificação:")
    print(f"  Baixa densidade : ≤ {p33:.0f} pessoas")
    print(f"  Moderada        : {p33:.0f}–{p66:.0f} pessoas")
    print(f"  Sobrelotação    : > {p66:.0f} pessoas")
    print("\nGuarda estes valores — vais precisar deles no preprocessing.py!")


if __name__ == "__main__":
    main()