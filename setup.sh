#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Kaggle API if not already installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle API..."
    pip install kaggle
    
    echo "Please set up your Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Scroll down to the 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Move the downloaded kaggle.json to ~/.kaggle/"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo "\nAfter setting up the credentials, run this script again."
    exit 1
fi

echo "Setup complete! You can now run the download script:"
echo "python src/download_dataset.py"
