import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from train_fixed import Config, TrafficSignDataset, load_model

def evaluate_model(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else range(43),
                yticklabels=class_names if class_names else range(43))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    return accuracy

def main():
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model - look for best_model.pth in the parent directory
    model_paths = [
        os.path.join('..', 'best_model.pth'),  # If running from src/
        'best_model.pth',                     # If running from project root
        os.path.join('data', 'models', 'best_model.pth')  # Alternative location
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading model from: {os.path.abspath(path)}")
            model = load_model(path, device)
            break
    
    if model is None:
        print("Error: Could not find model file in any of these locations:")
        for path in model_paths:
            print(f"  - {os.path.abspath(path)}")
        return
    
    # Data transforms (should match training)
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669])
        ])
    }
    
    # Load test data - handle paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, 'data', 'processed')
    labels_path = os.path.join(base_dir, 'labels.csv')
    
    print(f"Looking for labels at: {labels_path}")
    
    if not os.path.exists(labels_path):
        print(f"Error: Could not find {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Resolve paths for test set
    def resolve_paths(local_df, base_dir):
        def _resolve(row):
            img_path = str(row['image_path']).replace('\\', '/')
            filename = os.path.basename(img_path)
            split = str(row['split'])
            full = os.path.join(base_dir, 'images', split, filename)
            return str(full)
            
        local_df = local_df.copy()
        local_df['resolved_path'] = local_df.apply(_resolve, axis=1)
        exists_mask = local_df['resolved_path'].apply(lambda p: os.path.exists(p))
        removed = int((~exists_mask).sum())
        if removed:
            print(f"Dropping {removed} missing test files")
        return local_df[exists_mask].reset_index(drop=True)
    
    test_df = resolve_paths(test_df, base_dir)
    print(f"Test samples: {len(test_df)}")
    
    # Create test dataset and dataloader
    test_dataset = TrafficSignDataset(
        test_df, 
        transform=data_transforms['test'], 
        base_dir=base_dir
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=Config.num_workers
    )
    
    # Load class names if available
    class_names = None
    label_names_path = os.path.join(base_dir, 'label_names.csv')
    if os.path.exists(label_names_path):
        try:
            class_df = pd.read_csv(label_names_path)
            class_names = class_df['name'].tolist()
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Could not load class names: {e}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    accuracy = evaluate_model(model, test_loader, device, class_names)
    print(f"\nTest Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
