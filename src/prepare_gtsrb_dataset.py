import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image

def prepare_gtsrb_dataset(input_dir, output_dir, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
    
    Args:
        input_dir (Path): Path to the extracted GTSRB dataset
        output_dir (Path): Path to save the processed dataset
        val_ratio (float): Ratio of validation data (default: 0.15)
        test_ratio (float): Ratio of test data (default: 0.15)
    """
    # Define paths
    train_dir = input_dir / 'Train'
    test_dir = input_dir / 'Test'
    meta_dir = input_dir / 'Meta'
    
    # Create output directories
    output_train = output_dir / 'train'
    output_val = output_dir / 'val'
    output_test = output_dir / 'test'
    
    for dir_path in [output_train, output_val, output_test]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_csv = train_dir / 'Train.csv'
    
    if train_csv.exists():
        # New format with CSV
        df = pd.read_csv(train_csv)
        
        # Get unique classes
        classes = df['ClassId'].unique()
        
        # Create class directories
        for split in ['train', 'val']:
            for class_id in classes:
                (output_dir / split / str(class_id)).mkdir(parents=True, exist_ok=True)
        
        # Split into train/val
        for class_id in tqdm(classes, desc="Processing classes"):
            class_images = df[df['ClassId'] == class_id]
            
            # Shuffle and split
            class_images = class_images.sample(frac=1, random_state=42).reset_index(drop=True)
            val_size = int(len(class_images) * val_ratio)
            
            # Copy training images
            for _, row in class_images[val_size:].iterrows():
                src = train_dir / row['Path']
                dst = output_train / str(class_id) / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
            
            # Copy validation images
            for _, row in class_images[:val_size].iterrows():
                src = train_dir / row['Path']
                dst = output_val / str(class_id) / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
    else:
        # Old format with class folders
        class_dirs = [d for d in os.listdir(train_dir) if (train_dir / d).is_dir()]
        
        for class_id in tqdm(class_dirs, desc="Processing classes"):
            class_path = train_dir / class_id
            
            # Create class directories
            for split in ['train', 'val']:
                (output_dir / split / class_id).mkdir(parents=True, exist_ok=True)
            
            # Get all images
            image_files = list(class_path.glob('*.png')) + list(class_path.glob('*.jpg'))
            random.shuffle(image_files)
            
            # Split into train/val
            val_size = int(len(image_files) * val_ratio)
            
            # Copy training images
            for img in image_files[val_size:]:
                dst = output_train / class_id / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
            
            # Copy validation images
            for img in image_files[:val_size]:
                dst = output_val / class_id / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
    
    # Process test data
    print("\nProcessing test data...")
    test_csv = test_dir / 'Test.csv'
    
    if test_csv.exists():
        # New format with CSV
        df_test = pd.read_csv(test_csv)
        
        for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Copying test images"):
            src = test_dir / row['Path']
            class_id = str(row['ClassId'])
            
            # Create class directory if it doesn't exist
            (output_test / class_id).mkdir(parents=True, exist_ok=True)
            
            dst = output_test / class_id / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    else:
        # Old format with class folders
        for class_id in os.listdir(test_dir):
            class_path = test_dir / class_id
            if not class_path.is_dir():
                continue
                
            # Create class directory
            (output_test / class_id).mkdir(parents=True, exist_ok=True)
            
            # Copy test images
            for img in class_path.glob('*.png'):
                dst = output_test / class_id / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
    
    print(f"\nDataset prepared successfully at {output_dir}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in ['train', 'val', 'test']:
        split_path = output_dir / split
        num_classes = len([d for d in os.listdir(split_path) if (split_path / d).is_dir()])
        num_images = sum([len(files) for _, _, files in os.walk(split_path)])
        print(f"{split.capitalize()}: {num_images} images in {num_classes} classes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GTSRB dataset for training')
    parser.add_argument('--input_dir', type=str, default='../archive',
                       help='Path to the extracted GTSRB dataset')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Path to save the processed dataset')
    
    args = parser.parse_args()
    
    prepare_gtsrb_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )
