import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image 

def get_dataloaders(data_root='.', batch_size=64, image_size=128, num_workers=0):
    """
    Creates PyTorch DataLoaders for train and test sets from local image folders,
    assuming a 'data/train' and 'data/test' structure.
    Applies data augmentation to the training set.

    Returns:
        tuple: A tuple containing (train_loader, test_loader), or (None, None) if paths are invalid.
    """

    train_dir = os.path.join(data_root, 'data', 'train')
    test_dir = os.path.join(data_root, 'data', 'test')

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"Error: '{os.path.basename(train_dir)}' or '{os.path.basename(test_dir)}' folders not found inside '{os.path.join(data_root, 'data')}'")
        print("Please ensure your folder structure is 'data/train' and 'data/test'.")
        return None, None
    
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5), # On-the-fly data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
    ])
    
    # Testing transforms do not include augmentation to ensure consistent evaluation.
    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def is_valid_file(path):
        try:
            Image.open(path).verify()
            return True
        except:
            print(f"Warning: Skipping corrupted or invalid file: {path}")
            return False

    print(f"Loading training data from: {train_dir}")

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'data'), transform=train_transforms, is_valid_file=is_valid_file)
    train_indices = [i for i, (path, label) in enumerate(train_dataset.samples) if train_dataset.classes[label] == 'train']
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)


    print(f"Loading test data from: {test_dir}")
    test_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'data'), transform=test_transforms, is_valid_file=is_valid_file)
    test_indices = [i for i, (path, label) in enumerate(test_dataset.samples) if test_dataset.classes[label] == 'test']
    test_sampler = torch.utils.data.SequentialSampler(test_indices) # No shuffle for test set
    

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )
    
    print("\nDataLoaders created successfully!")
    print(f"Number of training images: {len(train_indices)}")
    print(f"Number of testing images: {len(test_indices)}")
    
    return train_loader, test_loader


