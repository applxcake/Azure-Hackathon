import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def process_pickle_file(pickle_path, output_dir, split_name):
    """Process a single pickle file and save images and labels."""
    # Create output directories
    img_dir = output_dir / 'images' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the pickle file
    data = load_pickle(pickle_path)
    
    # Extract features and labels
    images = data['features']
    labels = data['labels']
    
    print(f"Processing {split_name} set with {len(images)} images...")
    
    # Save images and collect label information
    label_info = []
    for i, (image, label) in enumerate(tqdm(zip(images, labels), total=len(images))):
        # Create filename
        img_filename = f"{split_name}_{i:05d}.png"
        img_path = img_dir / img_filename
        
        # Convert and save image
        img = Image.fromarray(image)
        img.save(img_path)
        
        # Save label info
        label_info.append({
            'image_path': str(img_path.relative_to(output_dir)),
            'label': int(label),
            'split': split_name
        })
    
    return label_info

def process_dataset(input_dir, output_dir):
    """Process the entire dataset."""
    # Convert to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ['train', 'valid', 'test']
    all_label_info = []
    
    for split in splits:
        pickle_path = input_dir / f"{split}.pickle"
        if not pickle_path.exists():
            print(f"Warning: {pickle_path} not found, skipping...")
            continue
            
        label_info = process_pickle_file(pickle_path, output_dir, split)
        all_label_info.extend(label_info)
    
    # Save label information to CSV
    df = pd.DataFrame(all_label_info)
    df.to_csv(output_dir / 'labels.csv', index=False)
    
    # Load and save label names
    label_names_path = input_dir / 'label_names.csv'
    if label_names_path.exists():
        shutil.copy(label_names_path, output_dir / 'label_names.csv')
    
    print("\nDataset processing complete!")
    print(f"Total images processed: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print(f"Output directory: {output_dir}")

def main():
    # Set paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / 'data' / 'raw'
    output_dir = base_dir / 'data' / 'processed'
    
    # Process the dataset
    process_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()
