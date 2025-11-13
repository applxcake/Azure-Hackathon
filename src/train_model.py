import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
class Config:
    data_dir = Path('data/processed')
    batch_size = 64
    num_epochs = 10
    num_workers = 4
    learning_rate = 0.001
    num_classes = 43
    input_size = 64  # Input image size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class TrafficSignDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = Config.data_dir / self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Model
class TrafficSignModel(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignModel, self).__init__()
        # Use a pretrained ResNet18 model
        self.model = models.resnet18(pretrained=True)
        
        # Replace the final fully connected layer
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

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training and validation loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it's the best so far
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save the best model
                torch.save(model.state_dict(), 'best_model.pth')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Evaluate function
def evaluate_model(model, dataloader, dataset_size, class_names):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = running_corrects.double() / dataset_size
    print(f'Test Accuracy: {acc:.4f}')
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Data loading and preprocessing
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
        ]),
        'test': transforms.Compose([
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load the dataset
    df = pd.read_csv(Config.data_dir / 'labels.csv')
    
    # Split into train, validation, and test sets
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    valid_df = df[df['split'] == 'valid'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Create datasets
    image_datasets = {
        'train': TrafficSignDataset(train_df, transform=data_transforms['train']),
        'valid': TrafficSignDataset(valid_df, transform=data_transforms['valid']),
        'test': TrafficSignDataset(test_df, transform=data_transforms['test'])
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=Config.batch_size, 
                          shuffle=True, num_workers=Config.num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=Config.batch_size, 
                          shuffle=False, num_workers=Config.num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=Config.batch_size, 
                         shuffle=False, num_workers=Config.num_workers)
    }
    
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'valid': len(image_datasets['valid']),
        'test': len(image_datasets['test'])
    }
    
    # Load class names
    class_names = pd.read_csv(Config.data_dir / 'label_names.csv')['SignName'].tolist()
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Number of classes: {len(class_names)}")
    
    # Initialize the model
    model = TrafficSignModel(num_classes=Config.num_classes)
    model = model.to(Config.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print("Starting training...")
    model = train_model(model, criterion, optimizer, scheduler, 
                       dataloaders, dataset_sizes, num_epochs=Config.num_epochs)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(model, dataloaders['test'], dataset_sizes['test'], class_names)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")

if __name__ == '__main__':
    main()
