import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def load_sample(img_path, gt_path):
    """Carrega uma imagem e o seu ground truth (.mat)"""
    img = np.array(Image.open(img_path).convert('RGB'))
    mat = sio.loadmat(gt_path)
    # O campo 'image_info' contém as coordenadas das pessoas
    locations = mat['image_info'][0][0][0][0][0]  # array (N, 2)
    count = len(locations)
    return img, locations, count

def load_split(base_path: str) -> list:
    """Carrega todas as imagens + ground truth de uma pasta (train ou test)."""
    base    = Path(base_path)
    img_dir = base / "images"
    gt_dir  = base / "ground-truth"
 
    if not img_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {img_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Pasta de ground truth não encontrada: {gt_dir}")
 
    samples = []
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
 
    for img_name in img_files:
        gt_name  = img_name.replace(".jpg", ".mat")
        img_path = img_dir / img_name
        gt_path  = gt_dir / f"GT_{gt_name}"
 
        if not gt_path.exists():
            print(f"  [AVISO] Ground truth não encontrada: {gt_path.name}")
            continue
 
        try:
            img, locs, count = load_sample(str(img_path), str(gt_path))
            samples.append((img, locs, count))
        except Exception as e:
            print(f"  [ERRO] {img_name}: {e}")
 
    print(f"  Carregadas {len(samples)} imagens de: {base_path}")
    return samples

def visualize_sample(img, locations, count):
    """Visualiza imagem com pontos de anotação"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Imagem original com pontos
    axes[0].imshow(img)
    axes[0].scatter(locations[:, 0], locations[:, 1], c='red', s=5, alpha=0.6)
    axes[0].set_title(f'Anotações — {count} pessoas')
    axes[0].axis('off')
    
    # Histograma de densidade 2D
    axes[1].hist2d(locations[:, 0], locations[:, 1], bins=50, cmap='hot')
    axes[1].set_title('Mapa de densidade (2D)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# testar se carregam se imagens
#(apagar depois)
if __name__ == "__main__":
    BASE = "data/ShanghaiTech/part_A/train_data"
 
    print("A carregar primeira imagem de Part A / train...")
    samples = load_split(BASE)
 
    if samples:
        img, locs, count = samples[0]
        print(f"  Shape da imagem : {img.shape}")
        print(f"  Nº de pessoas   : {count}")
        visualize_sample(img, locs, count)
    else:
        print("Nenhuma imagem carregada. Verifica o caminho BASE.")