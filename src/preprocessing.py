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

# Configurações Globais (defaults)
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


def extract_features(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Extrai vetor de features (HOG + LBP + Histograma de Cor)."""
    img_resized = resize(img, cfg["img_size"], anti_aliasing=True)
    img_gray    = rgb2gray(img_resized)

    # 1. HOG
    hog_features = hog(
        img_gray,
        orientations=cfg["hog_orientations"],
        pixels_per_cell=cfg["hog_pixels_per_cell"],
        cells_per_block=cfg["hog_cells_per_block"],
        visualize=False,
        feature_vector=True,
    )

    # 2. LBP
    n_points = 8 * cfg["lbp_radius"]
    lbp = local_binary_pattern(img_gray, n_points, cfg["lbp_radius"], method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=cfg["lbp_n_bins"], range=(0, cfg["lbp_n_bins"]), density=True)

    # 3. Histograma de cor RGB
    color_hist = []
    for channel in range(3):
        hist, _ = np.histogram(img_resized[:, :, channel], bins=32, range=(0, 1), density=True)
        color_hist.append(hist)
    color_hist = np.concatenate(color_hist)

    return np.concatenate([hog_features, lbp_hist, color_hist])


def process_samples(samples: list, cfg: dict, augment_data: bool = False) -> tuple:
    """Extrai features para uma lista de amostras, aplicando augmentation se solicitado."""
    X, y = [], []
    for img, _locations, count in tqdm(samples, desc="  A extrair features"):
        X.append(extract_features(img, cfg))
        y.append(count)

        if augment_data:
            for aug_img in augment(img):
                X.append(extract_features(aug_img, cfg))
                y.append(count)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_transform_pipeline(X_train: np.ndarray, X_test: np.ndarray, out_dir: Path, pca_variance: float) -> tuple:
    """Aplica Normalização e PCA mantendo os transformadores para uso posterior."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=pca_variance, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    with open(out_dir / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(out_dir / "pca_model.pkl", "wb") as f: pickle.dump(pca, f)

    return X_train_pca, X_test_pca, scaler, pca


def main():
    parser = argparse.ArgumentParser(description="Pipeline de pré-processamento")
    parser.add_argument("--data_root", type=str, default="./data/ShanghaiTech")
    parser.add_argument("--out_dir",   type=str, default="./processed")
    parser.add_argument("--name",      type=str, default=None,
                        help="Nome da subpasta de output (gerado automaticamente se omitido)")
    parser.add_argument("--part",      type=str, default="both", choices=["part_A", "part_B", "both"])
    parser.add_argument("--no_augment", action="store_true", help="Desativar data augmentation")
    # Parâmetros de feature extraction
    parser.add_argument("--img_size",            type=int,   nargs=2, default=[128, 128])
    parser.add_argument("--lbp_radius",          type=int,   default=3)
    parser.add_argument("--lbp_n_bins",          type=int,   default=64)
    parser.add_argument("--hog_orientations",    type=int,   default=9)
    parser.add_argument("--hog_pixels_per_cell", type=int,   nargs=2, default=[16, 16])
    parser.add_argument("--hog_cells_per_block", type=int,   nargs=2, default=[2, 2])
    parser.add_argument("--pca_variance",        type=float, default=0.95)
    args = parser.parse_args()

    cfg = {
        "img_size":            tuple(args.img_size),
        "lbp_radius":          args.lbp_radius,
        "lbp_n_bins":          args.lbp_n_bins,
        "hog_orientations":    args.hog_orientations,
        "hog_pixels_per_cell": tuple(args.hog_pixels_per_cell),
        "hog_cells_per_block": tuple(args.hog_cells_per_block),
        "pca_variance":        args.pca_variance,
    }

    # Nome da subpasta: explícito ou gerado a partir dos parâmetros
    if args.name:
        run_name = args.name
    else:
        run_name = (
            f"img{cfg['img_size'][0]}x{cfg['img_size'][1]}"
            f"_hog{cfg['hog_orientations']}o{cfg['hog_pixels_per_cell'][0]}p"
            f"_lbp{cfg['lbp_radius']}r{cfg['lbp_n_bins']}b"
            f"_pca{int(cfg['pca_variance']*100)}"
            f"{'_noaug' if args.no_augment else ''}"
        )

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = ["part_A", "part_B"] if args.part == "both" else [args.part]

    print(f"\n=== Pipeline de Pré-processamento ===")
    print(f"  Output: {out_dir}\n")

    # Carregamento de dados
    train_samples, test_samples = [], []
    for part in parts:
        train_samples += load_split(str(data_root / part / "train_data"))
        test_samples  += load_split(str(data_root / part / "test_data"))

    augment_data = not args.no_augment
    print(f"  Treino : {len(train_samples)} imagens originais")
    print(f"  Teste  : {len(test_samples)} imagens")

    # Extração de features
    X_train, y_reg_train = process_samples(train_samples, cfg, augment_data=augment_data)
    X_test,  y_reg_test  = process_samples(test_samples,  cfg, augment_data=False)

    # Normalização e PCA
    X_train_pca, X_test_pca, scaler, pca = fit_transform_pipeline(X_train, X_test, out_dir, cfg["pca_variance"])

    # Persistência de dados
    np.save(out_dir / "X_train.npy",     X_train_pca)
    np.save(out_dir / "X_test.npy",      X_test_pca)
    np.save(out_dir / "y_reg_train.npy", y_reg_train)
    np.save(out_dir / "y_reg_test.npy",  y_reg_test)

    # Metadados
    meta = {
        "run_name": run_name,
        "pca_n_components": int(pca.n_components_),
        "img_size": list(cfg["img_size"]),
        "lbp_radius": cfg["lbp_radius"],
        "lbp_n_bins": cfg["lbp_n_bins"],
        "hog_orientations": cfg["hog_orientations"],
        "hog_pixels_per_cell": list(cfg["hog_pixels_per_cell"]),
        "hog_cells_per_block": list(cfg["hog_cells_per_block"]),
        "pca_variance": cfg["pca_variance"],
        "augmentation": augment_data,
        "parts_used": parts,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== Pré-processamento concluído → {out_dir} ===")


if __name__ == "__main__":
    main()
