import os
import glob
from PIL import Image
from tqdm import tqdm
import argparse

def analyze_image_folder(image_dir, min_width=64, min_height=64):
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at '{os.path.abspath(image_dir)}'")
        return

    print(f"--- Analyzing Directory: {os.path.abspath(image_dir)} ---")

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
    all_potential_paths = []
    for ext in image_extensions:
        all_potential_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    total_files = len(all_potential_paths)
    if total_files == 0:
        print("No image files found in this directory.")
        return

    valid_files = 0
    for path in tqdm(all_potential_paths, desc="Checking image sizes"):
        try:
            with Image.open(path) as img:
                if img.width >= min_width and img.height >= min_height:
                    valid_files += 1
        except Exception:
            continue

    rejected_files = total_files - valid_files
    percentage_kept = (valid_files / total_files) * 100 if total_files > 0 else 0

    print(f"Total image files found:      {total_files}")
    print(f"Images meeting the minimum size requirement (>= {min_width}x{min_height}px):")
    print(f"  - Kept for training:        {valid_files}")
    print(f"  - Discarded (too small):    {rejected_files}")
    print(f"  - Percentage to be used:    {percentage_kept:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze an image folder to count total and valid-sized images.")
    parser.add_argument(
        "directory", 
        type=str, 
        help="Path to the image directory to analyze (e.g., 'data/train')."
    )
    
    args = parser.parse_args()
    
    analyze_image_folder(args.directory)

