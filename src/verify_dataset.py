import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random
from PIL import Image

def verify_dataset_structure(dataset_path):
    """Verify the dataset structure and count files."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found at {dataset_path}")
        return False
    
    print(f"\nDataset directory: {dataset_path}")
    
    # Check for common dataset structures
    possible_structures = {
        'class_folders': lambda p: any((p/d).is_dir() for d in os.listdir(p) if (p/d).is_dir()),
        'train_val_test': lambda p: all((p/part).is_dir() for part in ['train', 'val', 'test'])
    }
    
    structure_type = None
    for name, check in possible_structures.items():
        if check(dataset_path):
            structure_type = name
            break
    
    if structure_type == 'class_folders':
        print("Detected structure: Class folders")
        class_dirs = [d for d in os.listdir(dataset_path) if (dataset_path/d).is_dir()]
        print(f"Found {len(class_dirs)} classes")
        
        # Count images per class
        print("\nClass distribution:")
        for class_dir in sorted(class_dirs):
            class_path = dataset_path / class_dir
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            print(f"- {class_dir}: {len(images)} images")
        
        return True
        
    elif structure_type == 'train_val_test':
        print("Detected structure: Train/Val/Test split")
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            classes = [d for d in os.listdir(split_path) if (split_path/d).is_dir()]
            print(f"\n{split.capitalize()} set:")
            print(f"- Classes: {len(classes)}")
            
            # Count images in first few classes to verify
            sample_classes = classes[:3]  # Show first 3 classes as sample
            for class_name in sample_classes:
                class_path = split_path / class_name
                images = list(class_path.glob('*.[jJ][pP][gG]')) + list(class_path.glob('*.[pP][nN][gG]'))
                print(f"  - {class_name}: {len(images)} images")
            
            if len(classes) > 3:
                print(f"  - ... and {len(classes) - 3} more classes")
        
        return True
    
    else:
        print("Could not determine dataset structure. Please ensure your dataset has one of these structures:")
        print("1. Class folders: dataset/class1/, dataset/class2/, etc.")
        print("2. Train/Val/Test split: dataset/train/, dataset/val/, dataset/test/")
        return False

def prepare_dataset(input_dir, output_dir):
    """Prepare the dataset by creating train/val/test splits."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(input_dir) if (input_dir/d).is_dir()]
    
    if not class_dirs:
        print("No class directories found. Please check your dataset structure.")
        return False
    
    print(f"\nPreparing dataset with {len(class_dirs)} classes...")
    
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_path = input_dir / class_name
        
        # Get all images
        image_files = list(class_path.glob('*.[jJ][pP][gG]')) + list(class_path.glob('*.[pP][nN][gG]'))
        
        if not image_files:
            print(f"Warning: No images found in {class_path}")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Split into train (70%), val (15%), test (15%)
        n = len(image_files)
        train_end = int(0.7 * n)
        val_end = train_end + int(0.15 * n)
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Copy files to respective directories
        for split, files in splits.items():
            dest_dir = output_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for src_file in files:
                dest_file = dest_dir / src_file.name
                if not dest_file.exists():  # Skip if already exists
                    shutil.copy2(src_file, dest_file)
    
    print(f"\nDataset prepared successfully at {output_dir}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify and prepare traffic sign dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to the input dataset directory')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Path to the output directory (default: ../data/processed)')
    parser.add_argument('--prepare', action='store_true',
                       help='Prepare the dataset by creating train/val/test splits')
    
    args = parser.parse_args()
    
    # Verify dataset structure
    is_valid = verify_dataset_structure(Path(args.input_dir))
    
    # Prepare dataset if requested and structure is valid
    if is_valid and args.prepare:
        prepare_dataset(args.input_dir, args.output_dir)
