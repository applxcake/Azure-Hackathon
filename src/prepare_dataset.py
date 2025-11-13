import os
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm

def prepare_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Prepare the dataset by splitting it into train/validation/test sets.
    
    Args:
        input_dir (str): Path to the directory containing the dataset
        output_dir (str): Path to the output directory
        train_ratio (float): Ratio of training data (default: 0.8)
        val_ratio (float): Ratio of validation data (default: 0.1)
    """
    # Create output directories
    output_dirs = {
        'train': os.path.join(output_dir, 'train'),
        'val': os.path.join(output_dir, 'val'),
        'test': os.path.join(output_dir, 'test')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(class_dirs)} classes in the dataset")
    
    # Process each class
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_dir = os.path.join(input_dir, class_name)
        
        # Create class directories in each split
        for split in output_dirs.values():
            os.makedirs(os.path.join(split, class_name), exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle the files
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Copy files to respective directories
        for i, img_file in enumerate(image_files):
            src = os.path.join(class_dir, img_file)
            
            if i < train_end:
                dst = os.path.join(output_dirs['train'], class_name, img_file)
            elif i < val_end:
                dst = os.path.join(output_dirs['val'], class_name, img_file)
            else:
                dst = os.path.join(output_dirs['test'], class_name, img_file)
                
            if not os.path.exists(dst):  # Skip if already exists
                shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description='Prepare traffic sign dataset for training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to the input dataset directory')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Path to the output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation data (default: 0.1)')
    
    args = parser.parse_args()
    
    print(f"Preparing dataset from {args.input_dir}")
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
