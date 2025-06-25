import os
import glob
import shutil
from PIL import Image
from tqdm import tqdm
import argparse

def filter_and_copy_images(source_dir, dest_dir, min_width=64, min_height=64):
    """
    Scans a source directory, copies only the images that meet a minimum
    size requirement to a new destination directory.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Created destination directory: '{dest_dir}'")

    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
    all_potential_paths = []
    for ext in image_extensions:
        all_potential_paths.extend(glob.glob(os.path.join(source_dir, ext)))

    print(f"Found {len(all_potential_paths)} total files. Now filtering and copying...")

    copied_count = 0
    for path in tqdm(all_potential_paths, desc="Filtering and Copying"):
        try:
            with Image.open(path) as img:
                if img.width >= min_width and img.height >= min_height:
                    shutil.copy(path, dest_dir)
                    copied_count += 1
        except Exception:
            continue

    print("\n--- Filtering Complete ---")
    print(f"Successfully copied {copied_count} images (>= {min_width}x{min_height}px) to '{dest_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter a folder of images by size and copy them to a new location.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the source folder with all images.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path to the new folder to save the filtered images.")
    args = parser.parse_args()
    
    filter_and_copy_images(args.source_dir, args.dest_dir)