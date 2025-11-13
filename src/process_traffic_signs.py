import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

def process_traffic_signs(input_dir, output_dir):
    """Process the traffic signs dataset from pickle files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    splits = ['train', 'valid', 'test']
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    split_files = {
        'train': 'train.pickle',
        'valid': 'valid.pickle',
        'test': 'test.pickle'
    }
    
    for split, filename in split_files.items():
        print(f"Processing {split} data...")
        file_path = input_dir / filename
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict) or 'coords' not in data or 'labels' not in data:
                print(f"Unexpected data format in {filename}")
                continue
                
            coords = data['coords']
            labels = data['labels']
            
            print(f"Found {len(labels)} samples in {filename}")
            
            # Create a mapping from label to class name (if available)
            label_map = {}
            label_file = input_dir / 'label_names.csv'
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for i, line in enumerate(f):
                        label_map[i] = line.strip()
            else:
                # If no label file, just use numbers
                for label in np.unique(labels):
                    label_map[label] = str(label)
            
            # Process each image
            for i, (coord, label) in enumerate(zip(coords, labels)):
                # Create class directory
                class_dir = output_dir / split / str(label)
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Save coordinates to a text file
                coord_file = class_dir / f"{i:05d}.txt"
                with open(coord_file, 'w') as f:
                    f.write(','.join(map(str, coord)))
                
                # If there are images in the pickle file, they would be processed here
                # For now, we're just saving the coordinates
                
            print(f"Processed {len(labels)} samples for {split}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Save label mapping
    with open(output_dir / 'label_names.txt', 'w') as f:
        for label_id, label_name in sorted(label_map.items()):
            f.write(f"{label_id}: {label_name}\n")
    
    print(f"\nProcessing complete. Data saved to {output_dir}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in splits:
        split_path = output_dir / split
        if split_path.exists():
            num_classes = len([d for d in os.listdir(split_path) if (split_path / d).is_dir()])
            num_samples = sum([len(files) for _, _, files in os.walk(split_path)])
            print(f"{split.capitalize()}: {num_samples} samples in {num_classes} classes")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process traffic signs dataset')
    parser.add_argument('--input_dir', type=str, default='../archive',
                       help='Path to the input directory containing pickle files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Path to save the processed dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_traffic_signs(Path(args.input_dir), output_dir)
