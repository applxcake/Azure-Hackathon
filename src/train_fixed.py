import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import copy

# Configuration
class Config:
    data_dir = Path('data/processed')
    batch_size = 32  # Reduced for local training
    num_epochs = 5   # Reduced for testing
    num_workers = 2
    learning_rate = 0.001
    num_classes = 43
    input_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add data splits
    data_splits = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }

# Custom Dataset
class TrafficSignDataset(Dataset):
    def __init__(self, df, transform=None, base_dir='.'):
        self.df = df
        self.transform = transform
        self.base_dir = Path(base_dir)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # If a pre-resolved path column exists, use it directly
        if 'resolved_path' in self.df.columns:
            rp = self.df.iloc[idx]['resolved_path']
            if isinstance(rp, (str, Path)) and Path(rp).exists():
                full_path = Path(rp)
            else:
                full_path = None
        else:
            full_path = None

        img_path = self.df.iloc[idx]['image_path']
        split = str(self.df.iloc[idx]['split']) if 'split' in self.df.columns else ''

        # Normalize and extract filename
        if isinstance(img_path, str):
            img_path = img_path.replace('\\', '/')
            filename = os.path.basename(img_path)
        else:
            filename = str(img_path)

        # Build full path based on split
        if full_path is None and split in ('train', 'valid', 'test'):
            full_path = self.base_dir / 'images' / split / filename
        if full_path is None:
            # Fallback heuristics
            candidates = [
                self.base_dir / 'images' / 'train' / filename,
                self.base_dir / 'images' / 'valid' / filename,
                self.base_dir / 'images' / 'test' / filename,
                self.base_dir / filename,
                self.base_dir / 'images' / filename,
            ]
            full_path = None
            for c in candidates:
                if c.exists():
                    full_path = c
                    break
            if full_path is None:
                full_path = self.base_dir / 'images' / 'train' / filename
        
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return blank image if file not found
            image = Image.new('RGB', (64, 64), color=(0, 0, 0))
            label = 0
            if self.transform:
                image = self.transform(image)
            return image, label
            
        label = int(self.df.iloc[idx]['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Model
class TrafficSignModel(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Wrap dataloader with tqdm for progress bar
            dataloader = dataloaders[phase]
            if phase == 'train':
                dataloader = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} {phase}')
            
            for inputs, labels in dataloader:
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                if phase == 'train':
                    dataloader.set_postfix(loss=loss.item(), 
                                         acc=torch.sum(preds == labels.data).item()/len(labels))
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best model saved with accuracy: {best_acc:.4f}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    return model

def main():
    print("Starting training...")
    print(f"Using device: {Config.device}")
    
    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load the dataset
    base_dir = Path('data/processed')
    labels_path = base_dir / 'labels.csv'
    
    if not labels_path.exists():
        print(f"Error: Could not find {labels_path}")
        return
    
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} samples")
    
    # Split into train and validation sets
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    valid_df = df[df['split'] == 'valid'].reset_index(drop=True)

    print(f"Training samples (raw): {len(train_df)}")
    print(f"Validation samples (raw): {len(valid_df)}")

    # Resolve full paths and drop missing files to avoid I/O errors
    def resolve_paths(local_df: pd.DataFrame) -> pd.DataFrame:
        def _resolve(row):
            img_path = str(row['image_path']).replace('\\', '/')
            filename = os.path.basename(img_path)
            split = str(row['split'])
            full = base_dir / 'images' / split / filename
            return str(full)
        local_df = local_df.copy()
        local_df['resolved_path'] = local_df.apply(_resolve, axis=1)
        exists_mask = local_df['resolved_path'].apply(lambda p: Path(p).exists())
        removed = int((~exists_mask).sum())
        if removed:
            print(f"Dropping {removed} missing files from split '{local_df.iloc[0]['split']}'")
        return local_df[exists_mask].reset_index(drop=True)

    train_df = resolve_paths(train_df)
    valid_df = resolve_paths(valid_df)

    print(f"Training samples (kept): {len(train_df)}")
    print(f"Validation samples (kept): {len(valid_df)}")

    # Create datasets
    image_datasets = {
        'train': TrafficSignDataset(train_df, transform=data_transforms['train'], base_dir=base_dir),
        'valid': TrafficSignDataset(valid_df, transform=data_transforms['valid'], base_dir=base_dir)
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=Config.batch_size, 
                          shuffle=True, num_workers=Config.num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=Config.batch_size, 
                          shuffle=False, num_workers=Config.num_workers)
    }
    
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'valid': len(image_datasets['valid'])
    }
    
    # Initialize the model
    print("Initializing model...")
    model = TrafficSignModel(num_classes=Config.num_classes)
    model = model.to(Config.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print(f"Starting training for {Config.num_epochs} epochs...")
    model = train_model(model, criterion, optimizer, scheduler, 
                       dataloaders, dataset_sizes, num_epochs=Config.num_epochs)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete! Model saved as 'final_model.pth'")

def load_model(model_path, device, num_classes=43):
    """Load a trained model from disk."""
    model = TrafficSignModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

if __name__ == '__main__':
    main()
