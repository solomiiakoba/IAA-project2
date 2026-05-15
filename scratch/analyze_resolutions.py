import os
import numpy as np
from PIL import Image
from scipy import stats
import glob

def get_image_resolutions(directory):
    resolutions = []
    image_paths = glob.glob(os.path.join(directory, "*.jpg"))
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.size)  # (width, height)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
    return resolutions

def calculate_stats(resolutions, label):
    if not resolutions:
        print(f"No images found for {label}")
        return None
    
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    
    mean_w, mean_h = np.mean(widths), np.mean(heights)
    median_w, median_h = np.median(widths), np.median(heights)
    mode_w = stats.mode(widths, keepdims=True).mode[0]
    mode_h = stats.mode(heights, keepdims=True).mode[0]
    min_w, min_h = np.min(widths), np.min(heights)
    max_w, max_h = np.max(widths), np.max(heights)
    std_w, std_h = np.std(widths), np.std(heights)
    
    print(f"\n--- Statistics for {label} ---")
    print(f"Count: {len(resolutions)}")
    print(f"Mean Resolution: {mean_w:.2f} x {mean_h:.2f}")
    print(f"Median Resolution: {median_w} x {median_h}")
    print(f"Mode (Most Common): {mode_w} x {mode_h}")
    
    # Calculate top resolutions (peaks)
    res_counts = {}
    for r in resolutions:
        res_counts[r] = res_counts.get(r, 0) + 1
    sorted_res = sorted(res_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 Resolutions (Peaks):")
    for res, count in sorted_res[:5]:
        print(f"  - {res[0]}x{res[1]}: {count} images ({count/len(resolutions)*100:.1f}%)")
        
    print(f"Min Resolution: {min_w} x {min_h}")
    print(f"Max Resolution: {max_w} x {max_h}")
    print(f"Std Dev: {std_w:.2f} x {std_h:.2f}")
    
    return {
        'widths': widths,
        'heights': heights
    }

def main():
    base_path = "data/ShanghaiTech"
    parts = ['part_A', 'part_B']
    subdirs = ['train_data/images', 'test_data/images']
    
    all_res_A = []
    all_res_B = []
    
    for part in parts:
        part_res = []
        for subdir in subdirs:
            path = os.path.join(base_path, part, subdir)
            if os.path.exists(path):
                res = get_image_resolutions(path)
                part_res.extend(res)
            else:
                print(f"Directory not found: {path}")
        
        if part == 'part_A':
            all_res_A = part_res
            calculate_stats(all_res_A, "Part A")
        else:
            all_res_B = part_res
            calculate_stats(all_res_B, "Part B")
            
    all_res_combined = all_res_A + all_res_B
    calculate_stats(all_res_combined, "Combined (A + B)")

if __name__ == "__main__":
    main()
