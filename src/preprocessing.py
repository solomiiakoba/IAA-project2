"""
preprocessing.py — Pipeline de Pré-processamento
Augmentation, extração de features, normalização e PCA.
"""

import argparse
import json
import pickle
import warnings
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

# Supressão de avisos do LBP sobre imagens floating-point
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")

# Configurações Globais
IMG_SIZE             = (128, 128)
LBP_RADIUS           = 3
LBP_N_POINTS         = 8 * LBP_RADIUS
LBP_N_BINS           = 64
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (16, 16)
HOG_CELLS_PER_BLOCK  = (2, 2)
PCA_VARIANCE         = 0.95


def augment(img: np.ndarray) -> list:
    """Aplica transformações para aumentar o dataset de treino."""
    pil_img = Image.fromarray(img)
    augmented = [
        np.array(pil_img.transpose(Image.FLIP_LEFT_RIGHT)),
        np.array(pil_img.rotate(10,  expand=False, fillcolor=(0, 0, 0))),
        np.array(pil_img.rotate(-10, expand=False, fillcolor=(0, 0, 0))),
        np.array(ImageEnhance.Brightness(pil_img).enhance(0.7)),
        np.array(ImageEnhance.Brightness(pil_img).enhance(1.3))
    ]
    return augmented


def extract_features(img: np.ndarray) -> np.ndarray:
    """Extrai vetor de features (HOG + LBP + Histograma de Cor)."""
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
    img_gray    = rgb2gray(img_resized)

    # 1. HOG
    hog_features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True,
    )

    # 2. LBP
    lbp = local_binary_pattern(img_gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0, LBP_N_BINS), density=True)

    # 3. Histograma de cor RGB
    color_hist = []
    for channel in range(3):
        hist, _ = np.histogram(img_resized[:, :, channel], bins=32, range=(0, 1), density=True)
        color_hist.append(hist)
    color_hist = np.concatenate(color_hist)

    return np.concatenate([hog_features, lbp_hist, color_hist])


def process_samples(samples: list, augment_data: bool = False) -> tuple:
    """Extrai features para uma lista de amostras, aplicando augmentation se solicitado."""
    X, y = [], []
    for img, _locations, count in tqdm(samples, desc="  A extrair features"):
        X.append(extract_features(img))
        y.append(count)

        if augment_data:
            for aug_img in augment(img):
                X.append(extract_features(aug_img))
                y.append(count)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_classification_labels(y_reg: np.ndarray, p33: float, p66: float) -> np.ndarray:
    """Converte contagens contínuas em classes discretas."""
    y_clf = np.zeros(len(y_reg), dtype=np.int32)
    y_clf[(y_reg > p33) & (y_reg <= p66)] = 1
    y_clf[y_reg > p66] = 2
    return y_clf


def fit_transform_pipeline(X_train: np.ndarray, X_test: np.ndarray, out_dir: Path) -> tuple:
    """Aplica Normalização e PCA mantendo os transformadores para uso posterior."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    with open(out_dir / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(out_dir / "pca_model.pkl", "wb") as f: pickle.dump(pca, f)

    return X_train_pca, X_test_pca, scaler, pca


def main():
    parser = argparse.ArgumentParser(description="Pipeline de pré-processamento")
    parser.add_argument("--data_root", type=str, default="./data/ShanghaiTech")
    parser.add_argument("--out_dir",   type=str, default="./processed")
    parser.add_argument("--part",      type=str, default="both", choices=["part_A", "part_B", "both"])
    parser.add_argument("--p33", type=float, default=None, help="Limiar inferior")
    parser.add_argument("--p66", type=float, default=None, help="Limiar superior")
    parser.add_argument("--no_augment", action="store_true", help="Desativar data augmentation")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = ["part_A", "part_B"] if args.part == "both" else [args.part]

    print("\n=== Pipeline de Pré-processamento ===\n")

    # Carregamento de dados
    train_samples, test_samples = [], []
    for part in parts:
        train_samples += load_split(str(data_root / part / "train_data"))
        test_samples  += load_split(str(data_root / part / "test_data"))

    augment_data = not args.no_augment
    print(f"  Treino : {len(train_samples)} imagens originais")
    print(f"  Teste  : {len(test_samples)} imagens")

    # Extração de features
    X_train, y_reg_train = process_samples(train_samples, augment_data=augment_data)
    X_test,  y_reg_test  = process_samples(test_samples,  augment_data=False)

    # Definição de limiares
    p33 = args.p33 if args.p33 is not None else float(np.percentile(y_reg_train, 33))
    p66 = args.p66 if args.p66 is not None else float(np.percentile(y_reg_train, 66))

    # Labels de classificação
    y_clf_train = make_classification_labels(y_reg_train, p33, p66)
    y_clf_test  = make_classification_labels(y_reg_test,  p33, p66)

    # Normalização e PCA
    X_train_pca, X_test_pca, scaler, pca = fit_transform_pipeline(X_train, X_test, out_dir)

    # Persistência de dados
    np.save(out_dir / "X_train.npy",     X_train_pca)
    np.save(out_dir / "X_test.npy",      X_test_pca)
    np.save(out_dir / "y_reg_train.npy", y_reg_train)
    np.save(out_dir / "y_reg_test.npy",  y_reg_test)
    np.save(out_dir / "y_clf_train.npy", y_clf_train)
    np.save(out_dir / "y_clf_test.npy",  y_clf_test)

    # Metadados
    meta = {
        "p33": p33, "p66": p66,
        "pca_n_components": int(pca.n_components_),
        "img_size": list(IMG_SIZE),
        "augmentation": augment_data
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Pré-processamento concluído ===")


if __name__ == "__main__":
    main()