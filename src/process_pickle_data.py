import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def process_pickle_data(input_dir, output_dir):
    """Process pickle files and extract images."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    splits = ['train', 'valid', 'test']
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load label names
    label_names = []
    with open(input_dir / 'label_names.csv', 'r') as f:
        label_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Process each split
    split_files = {
        'train': input_dir / 'train.pickle',
        'valid': input_dir / 'valid.pickle',
        'test': input_dir / 'test.pickle'
    }
    
    for split, file_path in split_files.items():
        print(f"Processing {split} data...")
        data = load_pickle(file_path)
        
        # The structure might vary, so we need to handle different cases
        if isinstance(data, dict):
            images = data.get(b'data', data.get('data', []))
            labels = data.get(b'labels', data.get('labels', []))
        elif isinstance(data, tuple) and len(data) == 2:
            images, labels = data
        else:
            print(f"Unexpected data format in {file_path}")
            continue
        
        # Convert to numpy arrays if they aren't already
        images = np.array(images)
        labels = np.array(labels)
        
        # Reshape images if needed (assuming they're flattened)
        if len(images.shape) == 2:
            # Assuming square images, calculate size
            size = int(np.sqrt(images.shape[1] / 3))  # Assuming RGB
            images = images.reshape(-1, size, size, 3).astype(np.uint8)
        
        # Save images
        for i, (img, label) in enumerate(zip(images, labels)):
            # Create class directory
            class_dir = output_dir / split / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            img = Image.fromarray(img)
            img_path = class_dir / f"{i:05d}.png"
            img.save(img_path)
    
    # Save label names
    with open(output_dir / 'label_names.txt', 'w') as f:
        for i, name in enumerate(label_names):
            f.write(f"{i} {name}\n")
    
    print(f"\nDataset processed successfully at {output_dir}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in splits:
        split_path = output_dir / split
        if split_path.exists():
            num_classes = len([d for d in os.listdir(split_path) if (split_path / d).is_dir()])
            num_images = sum([len(files) for _, _, files in os.walk(split_path)])
            print(f"{split.capitalize()}: {num_images} images in {num_classes} classes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process pickle dataset')
    parser.add_argument('--input_dir', type=str, default='../archive',
                       help='Path to the input directory containing pickle files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Path to save the processed dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_pickle_data(Path(args.input_dir), output_dir)
