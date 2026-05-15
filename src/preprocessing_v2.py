"""
preprocessing_v2.py — Pipeline de Pré-processamento Evoluído
Suporta múltiplas resoluções, técnicas de redimensionamento (com/sem padding) e configuração de HOG.
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

# Supressão de avisos do LBP
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")

# Configurações Fixas
LBP_RADIUS           = 3
LBP_N_POINTS         = 8 * LBP_RADIUS
LBP_N_BINS           = 64
HOG_ORIENTATIONS     = 9
PCA_VARIANCE         = 0.95

def resize_with_padding(img_np: np.ndarray, target_size: tuple) -> np.ndarray:
    """Redimensiona preservando o rácio de aspeto e adicionando padding (preto)."""
    img = Image.fromarray((img_np * 255).astype(np.uint8)) if img_np.dtype != np.uint8 else Image.fromarray(img_np)
    
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
        
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(resized_img, (paste_x, paste_y))
    
    return np.array(new_img)

def augment(img: np.ndarray) -> list:
    pil_img = Image.fromarray(img)
    augmented = [
        np.array(pil_img.transpose(Image.FLIP_LEFT_RIGHT)),
        np.array(pil_img.rotate(10,  expand=False, fillcolor=(0, 0, 0))),
        np.array(pil_img.rotate(-10, expand=False, fillcolor=(0, 0, 0))),
        np.array(ImageEnhance.Brightness(pil_img).enhance(0.7)),
        np.array(ImageEnhance.Brightness(pil_img).enhance(1.3))
    ]
    return augmented

def extract_features(img: np.ndarray, target_size: tuple, use_padding: bool, hog_ppc: tuple, hog_cpb: tuple) -> np.ndarray:
    if use_padding:
        img_resized = resize_with_padding(img, target_size)
        img_resized = img_resized.astype(np.float32) / 255.0
    else:
        img_resized = resize(img, (target_size[1], target_size[0]), anti_aliasing=True)

    img_gray = rgb2gray(img_resized)

    # 1. HOG
    hog_features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=hog_ppc,
        cells_per_block=hog_cpb,
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

def process_samples(samples: list, target_size: tuple, use_padding: bool, hog_ppc: tuple, hog_cpb: tuple, augment_data: bool = False) -> tuple:
    X, y = [], []
    for img, _locations, count in tqdm(samples, desc="  A extrair features"):
        X.append(extract_features(img, target_size, use_padding, hog_ppc, hog_cpb))
        y.append(count)

        if augment_data:
            for aug_img in augment(img):
                X.append(extract_features(aug_img, target_size, use_padding, hog_ppc, hog_cpb))
                y.append(count)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def make_classification_labels(y_reg: np.ndarray, p33: float, p66: float) -> np.ndarray:
    y_clf = np.zeros(len(y_reg), dtype=np.int32)
    y_clf[(y_reg > p33) & (y_reg <= p66)] = 1
    y_clf[y_reg > p66] = 2
    return y_clf

def fit_transform_pipeline(X_train: np.ndarray, X_test: np.ndarray, out_dir: Path) -> tuple:
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
    parser = argparse.ArgumentParser(description="Pipeline de pré-processamento V2")
    parser.add_argument("--data_root", type=str, default="./data/ShanghaiTech")
    parser.add_argument("--out_base",  type=str, default="./processed")
    parser.add_argument("--part",      type=str, default="both", choices=["part_A", "part_B", "both"])
    parser.add_argument("--width",     type=int, default=128)
    parser.add_argument("--height",    type=int, default=128)
    parser.add_argument("--padding",   action="store_true", help="Usar padding para preservar rácio")
    parser.add_argument("--hog_ppc",   type=int, default=16, help="HOG Pixels per cell")
    parser.add_argument("--hog_cpb",   type=int, default=2,  help="HOG Cells per block")
    parser.add_argument("--p33", type=float, default=None)
    parser.add_argument("--p66", type=float, default=None)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    target_size = (args.width, args.height)
    hog_ppc = (args.hog_ppc, args.hog_ppc)
    hog_cpb = (args.hog_cpb, args.hog_cpb)
    
    method_suffix = "pad" if args.padding else "resize"
    out_dir_name = f"proc_{args.width}x{args.height}_{method_suffix}_hog{args.hog_ppc}_{args.part}"
    out_dir = Path(args.out_base) / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    parts = ["part_A", "part_B"] if args.part == "both" else [args.part]

    print(f"\n=== Pipeline V2: {args.width}x{args.height} ({method_suffix}) | HOG PPC: {args.hog_ppc} ===\n")

    train_samples, test_samples = [], []
    for part in parts:
        train_samples += load_split(str(data_root / part / "train_data"))
        test_samples  += load_split(str(data_root / part / "test_data"))

    X_train, y_reg_train = process_samples(train_samples, target_size, args.padding, hog_ppc, hog_cpb, not args.no_augment)
    X_test,  y_reg_test  = process_samples(test_samples,  target_size, args.padding, hog_ppc, hog_cpb, False)

    p33 = args.p33 if args.p33 is not None else float(np.percentile(y_reg_train, 33))
    p66 = args.p66 if args.p66 is not None else float(np.percentile(y_reg_train, 66))

    y_clf_train = make_classification_labels(y_reg_train, p33, p66)
    y_clf_test  = make_classification_labels(y_reg_test,  p33, p66)

    X_train_pca, X_test_pca, scaler, pca = fit_transform_pipeline(X_train, X_test, out_dir)

    np.save(out_dir / "X_train.npy",     X_train_pca)
    np.save(out_dir / "X_test.npy",      X_test_pca)
    np.save(out_dir / "y_reg_train.npy", y_reg_train)
    np.save(out_dir / "y_reg_test.npy",  y_reg_test)
    np.save(out_dir / "y_clf_train.npy", y_clf_train)
    np.save(out_dir / "y_clf_test.npy",  y_clf_test)

    meta = {
        "p33": p33, "p66": p66,
        "pca_n_components": int(pca.n_components_),
        "img_size": [args.width, args.height],
        "use_padding": args.padding,
        "hog_ppc": args.hog_ppc,
        "part": args.part
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nConcluído! Dados em: {out_dir}")

if __name__ == "__main__":
    main()
