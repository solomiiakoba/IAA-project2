"""
Pipeline de Pré-processamento
Augmentation, extração de features, normalização e PCA.

Uso:
    python preprocessing.py --data_root ./data/ShanghaiTech \
                             --out_dir ./processed \
                             --p33 102 --p66 248

    (usar os valores de p33 e p66 que o eda.py imprimiu no terminal)

Gera:
    processed/
        X_train.npy — features de treino  (n_amostras, n_features)
        X_test.npy — features de teste
        y_reg_train.npy — contagens reais (regressão)
        y_reg_test.npy
        y_clf_train.npy — classes 0/1/2  (classificação)
        y_clf_test.npy
        pca_model.pkl — PCA ajustado (para usar nos modelos)
        scaler.pkl — StandardScaler ajustado
        meta.json — limiares, n_components, shape das features
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import load_split 


# Configuração
IMG_SIZE             = (128, 128)
LBP_RADIUS           = 3
LBP_N_POINTS         = 8 * LBP_RADIUS
LBP_N_BINS           = 64
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (16, 16)
HOG_CELLS_PER_BLOCK  = (2, 2)
PCA_VARIANCE         = 0.95     # manter 95% da variância


# Data Augmentation
def augment(img: np.ndarray) -> list:
    """
    Gera variações de uma imagem para aumentar o dataset de treino.
    Transformações: flip horizontal, rotação leve (+10°, -10°), brilho reduzido (simula pouca luz), brilho aumentado (simula sobreexposição)
    """
    pil_img = Image.fromarray(img)
    augmented = []

    augmented.append(np.array(pil_img.transpose(Image.FLIP_LEFT_RIGHT)))
    augmented.append(np.array(pil_img.rotate(10,  expand=False, fillcolor=(0, 0, 0))))
    augmented.append(np.array(pil_img.rotate(-10, expand=False, fillcolor=(0, 0, 0))))
    augmented.append(np.array(ImageEnhance.Brightness(pil_img).enhance(0.7)))
    augmented.append(np.array(ImageEnhance.Brightness(pil_img).enhance(1.3)))

    return augmented


#Extração de Features
def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extrai um vetor de features de uma imagem RGB. Combina HOG, LBP, Hist
    """
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
    img_gray    = rgb2gray(img_resized)

    # HOG
    hog_features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True,
    )

    # LBP
    lbp = local_binary_pattern(img_gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS,
                                range=(0, LBP_N_BINS), density=True)

    # Histograma de cor RGB
    color_hist = []
    for channel in range(3):
        hist, _ = np.histogram(img_resized[:, :, channel], bins=32,
                                range=(0, 1), density=True)
        color_hist.append(hist)
    color_hist = np.concatenate(color_hist)

    return np.concatenate([hog_features, lbp_hist, color_hist])


def process_samples(samples: list, augment_data: bool = False) -> tuple:
    """
    Se augment_data=True, aplica data augmentation (apenas para treino).
    """
    X, y = [], []

    for img, _locations, count in tqdm(samples, desc="  A extrair features"):
        X.append(extract_features(img))
        y.append(count)

        if augment_data:
            for aug_img in augment(img):
                X.append(extract_features(aug_img))
                y.append(count)   # a contagem mantém-se após augmentation

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# 3. Labels de Classificação
def make_classification_labels(y_reg: np.ndarray,
                                p33: float, p66: float) -> np.ndarray:
    """
    Converte contagens continuas em classes discretas:
        0 — Baixa densidade     (≤ p33)
        1 — Densidade moderada  (p33 < x ≤ p66)
        2 — Sobrelotação        (> p66)
    """
    y_clf = np.zeros(len(y_reg), dtype=np.int32)
    y_clf[(y_reg > p33) & (y_reg <= p66)] = 1
    y_clf[y_reg > p66] = 2
    return y_clf


# 4. Normalização + PCA
def fit_transform_pipeline(X_train: np.ndarray, X_test: np.ndarray, out_dir: Path) -> tuple:
    """
    StandardScaler - normaliza cada feature (média 0, std 1)
    PCA - reduz dimensionalidade mantendo 95% da variância
    Ajustar APENAS no treino, aplica ao teste (sem data leakage).
    Guarda scaler e PCA em disco para reutilizar nos modelos.
    """
    print(f"\n  Shape antes de PCA : {X_train.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    print(f"  Shape depois de PCA : {X_train_pca.shape}")
    print(f"  Componentes retidas : {pca.n_components_}")
    print(f"  Variância explicada : {pca.explained_variance_ratio_.sum():.3f}")

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    return X_train_pca, X_test_pca, scaler, pca

def main():
    parser = argparse.ArgumentParser(description="Pipeline de pré-processamento")
    parser.add_argument("--data_root", type=str, default="./data/ShanghaiTech")
    parser.add_argument("--out_dir",   type=str, default="./processed")
    parser.add_argument("--part",      type=str, default="both",
                        choices=["part_A", "part_B", "both"],
                        help="Qual parte do dataset usar")
    parser.add_argument("--p33", type=float, default=None,
                        help="Limiar inferior (percentil 33) do eda.py")
    parser.add_argument("--p66", type=float, default=None,
                        help="Limiar superior (percentil 66) do eda.py")
    parser.add_argument("--no_augment", action="store_true",
                        help="Desativar data augmentation")
    args = parser.parse_args()

    data_root    = Path(args.data_root)
    out_dir      = Path(args.out_dir)
    augment_data = not args.no_augment
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = ["part_A", "part_B"] if args.part == "both" else [args.part]

    print("\n=== Pipeline de Pré-processamento ===\n")

    # Carregar dados 
    print("A carregar imagens e ground truth...")
    train_samples, test_samples = [], []

    for part in parts:
        train_samples += load_split(str(data_root / part / "train_data"))
        test_samples  += load_split(str(data_root / part / "test_data"))

    print(f"\n  Treino : {len(train_samples)} imagens originais")
    if augment_data:
        print(f"           → {len(train_samples) * 6} após augmentation (×6)")
    print(f"  Teste  : {len(test_samples)} imagens\n")

    # Extrair features
    print("A extrair features (HOG + LBP + Histograma cor)...")
    print("  [Treino]")
    X_train, y_reg_train = process_samples(train_samples, augment_data=augment_data)
    print("  [Teste]")
    X_test,  y_reg_test  = process_samples(test_samples,  augment_data=False)

    # limiares 
    if args.p33 is None or args.p66 is None:
        print("\n  [AVISO] p33/p66 não fornecidos — a calcular a partir do treino.")
        print("          Recomenda-se usar os valores do eda.py.\n")
        p33 = float(np.percentile(y_reg_train, 33))
        p66 = float(np.percentile(y_reg_train, 66))
    else:
        p33, p66 = args.p33, args.p66

    print(f"  Limiares: baixa ≤ {p33:.0f} | moderada ≤ {p66:.0f} | sobrelotação > {p66:.0f}")

    # Labels de classificação
    y_clf_train = make_classification_labels(y_reg_train, p33, p66)
    y_clf_test  = make_classification_labels(y_reg_test,  p33, p66)

    print(f"\n  Distribuição classes (treino): {dict(enumerate(np.bincount(y_clf_train)))}")
    print(f"  Distribuição classes (teste) : {dict(enumerate(np.bincount(y_clf_test)))}")

    # Normalização + PCA
    print("\nA normalizar e reduzir dimensionalidade...")
    X_train_pca, X_test_pca, scaler, pca = fit_transform_pipeline(
        X_train, X_test, out_dir
    )

    # Guardar arrays
    print("\nA guardar ficheiros processados...")
    np.save(out_dir / "X_train.npy",     X_train_pca)
    np.save(out_dir / "X_test.npy",      X_test_pca)
    np.save(out_dir / "y_reg_train.npy", y_reg_train)
    np.save(out_dir / "y_reg_test.npy",  y_reg_test)
    np.save(out_dir / "y_clf_train.npy", y_clf_train)
    np.save(out_dir / "y_clf_test.npy",  y_clf_test)

    # Metadados
    meta = {
        "p33": p33,
        "p66": p66,
        "classes": {
            "0": f"Baixa densidade (≤ {p33:.0f} pessoas)",
            "1": f"Densidade moderada ({p33:.0f}–{p66:.0f} pessoas)",
            "2": f"Sobrelotação (> {p66:.0f} pessoas)",
        },
        "pca_n_components": int(pca.n_components_),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "img_size": list(IMG_SIZE),
        "augmentation": augment_data,
        "n_train_original": len(train_samples),
        "n_train_after_aug": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_shape_before_pca": list(X_train.shape),
        "feature_shape_after_pca":  list(X_train_pca.shape),
        "parts_used": parts,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Pré-processamento concluído ===")
    print(f"Ficheiros em: {out_dir.resolve()}")

if __name__ == "__main__":
    main()