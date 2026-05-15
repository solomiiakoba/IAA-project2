import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

def resize_image(img, target_size):
    """Simple resize (might distort)."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def pad_image(img, target_size):
    """Resize with padding to maintain aspect ratio."""
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
    return new_img

def main():
    # Define configurations including squares and rectangles
    resolutions = [
        (1024, 768), # Mode/Median
        (961, 696),  # Combined Mean
        (868, 589),  # Part A Mean
        (512, 384),  # Standard Rectangle 4:3
        (256, 192),  # Low-res Rectangle 4:3
        (128, 128),  # Square Baseline
        (64, 64),    # Square Baseline
        (128, 96),   # Small Rectangle
        (64, 48)     # Tiny Rectangle
    ]
    
    res_names = [f"Res_{r[0]}x{r[1]}" for r in resolutions]
    methods = ["resize", "pad"]
    
    samples = [
        ("data/ShanghaiTech/part_A/train_data/images/IMG_127.jpg", "PartA_ExtremeRes"), # Original is small
        ("data/ShanghaiTech/part_A/train_data/images/IMG_1.jpg", "PartA_Large"),
        ("data/ShanghaiTech/part_B/train_data/images/IMG_1.jpg", "PartB_Standard")
    ]
    
    output_dir = "eda_output/preprocessing_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, sample_name in samples:
        if not os.path.exists(img_path):
            continue
            
        orig_img = Image.open(img_path)
        print(f"Processing {sample_name} (Original: {orig_img.size})...")
        
        # Split into multiple comparison plots if too many
        cols = 2
        rows = len(resolutions)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
        plt.suptitle(f"Preprocessing Comparison: {sample_name} (Orig: {orig_img.size})", fontsize=16)
        
        for i, (res, res_name) in enumerate(zip(resolutions, res_names)):
            for j, method in enumerate(methods):
                ax = axes[i, j]
                if method == "resize":
                    processed = resize_image(orig_img, res)
                else:
                    processed = pad_image(orig_img, res)
                
                ax.imshow(processed)
                ax.set_title(f"{res_name} - {method.capitalize()}")
                ax.axis('off')
                
                # Save individual file
                save_name = f"{sample_name}_{res_name}_{method}.jpg"
                processed.save(os.path.join(output_dir, save_name))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_path = os.path.join(output_dir, f"{sample_name}_full_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    main()
