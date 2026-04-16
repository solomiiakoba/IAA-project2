import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

def load_sample(img_path, gt_path):
    """Carrega uma imagem e o seu ground truth (.mat)"""
    img = np.array(Image.open(img_path).convert('RGB'))
    mat = sio.loadmat(gt_path)
    # O campo 'image_info' contém as coordenadas das pessoas
    locations = mat['image_info'][0][0][0][0][0]  # array (N, 2)
    count = len(locations)
    return img, locations, count

def visualize_sample(img, locations, count):
    """Visualiza imagem com pontos de anotação"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Imagem original com pontos
    axes[0].imshow(img)
    axes[0].scatter(locations[:, 0], locations[:, 1], 
                    c='red', s=5, alpha=0.6)
    axes[0].set_title(f'Anotações — {count} pessoas')
    axes[0].axis('off')
    
    # Histograma de densidade
    axes[1].hist2d(locations[:, 0], locations[:, 1], 
                   bins=50, cmap='hot')
    axes[1].set_title('Mapa de densidade (2D)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# testar se carregam se imagens
if __name__ == "__main__":
    BASE = "data/ShanghaiTech/part_A/train_data"
    
    img_files = sorted(os.listdir(f"{BASE}/images"))
    
    # Carrega primeira imagem
    img_name = img_files[0]
    gt_name  = img_name.replace('.jpg', '.mat')
    
    img, locs, count = load_sample(
        f"{BASE}/images/{img_name}",
        f"{BASE}/ground-truth/GT_{gt_name}"
    )
    
    print(f"Imagem: {img.shape}")
    print(f"Pessoas: {count}")
    visualize_sample(img, locs, count)