import os
import zipfile
import shutil
from pathlib import Path
import kaggle

def setup_kaggle():
    """Set up Kaggle API credentials if they don't exist."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Kaggle API credentials not found.")
        print("Please follow these steps to set up Kaggle API:")
        print("1. Go to your Kaggle account settings")
        print("2. Scroll to the 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move the downloaded kaggle.json to ~/.kaggle/")
        print("\nAfter setting up the credentials, run this script again.")
        return False
    return True

def download_dataset():
    """Download and extract the dataset using Kaggle API."""
    # Create directories
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    for directory in [data_dir, raw_dir, processed_dir]:
        directory.mkdir(exist_ok=True)
    
    # Download the dataset
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        'valentynsichkar/traffic-signs-preprocessed',
        path=str(raw_dir),
        unzip=True
    )
    
    print("Dataset downloaded and extracted successfully!")
    return raw_dir, processed_dir

def process_dataset(raw_dir, processed_dir):
    """Process the dataset into the required format."""
    print("Processing dataset...")
    
    # The dataset contains .p files (pickle files)
    import pickle
    import numpy as np
    from PIL import Image
    
    # Process each pickle file
    for split in ['train', 'valid', 'test']:
        pickle_file = raw_dir / f'{split}.p'
        
        if not pickle_file.exists():
            print(f"Warning: {pickle_file} not found")
            continue
            
        # Load the pickle file
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        
        # Create output directory for this split
        split_dir = processed_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Process images and labels
        images = data['features']
        labels = data['labels']
        
        # Create subdirectories for each class
        for label in set(labels):
            (split_dir / str(label)).mkdir(exist_ok=True)
        
        # Save images
        for i, (image, label) in enumerate(zip(images, labels)):
            # Convert to PIL Image and save
            img = Image.fromarray(image)
            img_path = split_dir / str(label) / f'img_{i:05d}.png'
            img.save(img_path)
        
        print(f"Processed {len(images)} images for {split} set")

if __name__ == "__main__":
    if not setup_kaggle():
        exit(1)
    
    raw_dir, processed_dir = download_dataset()
    process_dataset(raw_dir, processed_dir)
