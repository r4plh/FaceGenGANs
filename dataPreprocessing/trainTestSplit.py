import os
import glob
import random
import shutil
import argparse
from tqdm import tqdm

def create_train_test_split(source_dir, dest_dir, train_count=6000):
    """
    Randomly splits image files from a source directory into 'train' and 'test' sets
    and moves them to a 'data/train' and 'data/test' structure within the destination folder.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{os.path.abspath(source_dir)}'")
        return

    print(f"Source directory:      '{os.path.abspath(source_dir)}'")
    print(f"Destination directory: '{os.path.abspath(dest_dir)}'")

    data_folder_path = os.path.join(dest_dir, 'data')
    os.makedirs(data_folder_path, exist_ok=True)
    
    train_path = os.path.join(data_folder_path, 'train')
    test_path = os.path.join(data_folder_path, 'test')
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)


    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(glob.glob(os.path.join(source_dir, ext)))
    
    total_images = len(all_image_files)
    print(f"Found {total_images} total images in the source directory.")

    if total_images < train_count:
        print(f"\nError: Total images found ({total_images}) is less than the requested training count ({train_count}).")
        print("Please check your source folder or lower the --train_count value.")
        return


    random.shuffle(all_image_files)
    print("Randomly shuffled all image files.")


    train_files = all_image_files[:train_count]
    test_files = all_image_files[train_count:]


    print(f"\nMoving {len(train_files)} images to '{os.path.join('data', 'train')}' folder...")
    for file_path in tqdm(train_files, desc="Moving train files"):
        filename = os.path.basename(file_path)
        dest_file_path = os.path.join(train_path, filename)
        shutil.move(file_path, dest_file_path)

    print(f"\nMoving {len(test_files)} images to '{os.path.join('data', 'test')}' folder...")
    for file_path in tqdm(test_files, desc="Moving test files"):
        filename = os.path.basename(file_path)
        dest_file_path = os.path.join(test_path, filename)
        shutil.move(file_path, dest_file_path)

    print("Splitting process completed successfully!")
    print(f"   - Training images: {len(os.listdir(train_path))}")
    print(f"   - Testing images:  {len(os.listdir(test_path))}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Randomly split a folder of images into a 'data/train' and 'data/test' structure.")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True, 
        help="Path to the folder containing all filtered images."
    )
    parser.add_argument(
        "--dest_dir", 
        type=str, 
        default='.', 
        help="Path to the folder where the 'data' directory will be created. Defaults to the current directory."
    )
    parser.add_argument(
        "--train_count", 
        type=int, 
        default=6000, 
        help="Number of images to put in the training set."
    )
    
    args = parser.parse_args()
    
    create_train_test_split(args.source_dir, args.dest_dir, args.train_count)
