# Traffic Sign Recognition

A deep learning model for recognizing and classifying traffic signs using PyTorch. This project demonstrates training a ResNet18 model on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## 📊 Model Performance

- **Test Accuracy**: 93.35%
- **Model Architecture**: ResNet18 with custom classification head
- **Input Size**: 224x224 pixels
- **Number of Classes**: 43

### Detailed Metrics

| Metric | Score |
|--------|-------|
| Precision (macro avg) | 0.88 |
| Recall (macro avg) | 0.87 |
| F1-score (macro avg) | 0.87 |
| Precision (weighted avg) | 0.94 |
| Recall (weighted avg) | 0.93 |
| F1-score (weighted avg) | 0.93 |

## 🚀 Features

- Data preprocessing and augmentation
- Model training with early stopping
- Learning rate scheduling
- Model evaluation with detailed metrics
- Confusion matrix visualization

## 🛠️ Project Structure

```
.
├── data/
│   └── processed/          # Processed dataset
│       ├── images/         # Extracted images
│       │   ├── train/      # Training images
│       │   ├── valid/      # Validation images
│       │   └── test/       # Test images
│       ├── labels.csv      # Image paths and labels
│       └── label_names.csv # Class names
├── src/
│   ├── train_fixed.py      # Training script
│   ├── evaluate.py         # Model evaluation
│   └── process_pickle_data.py # Data preprocessing
├── best_model.pth         # Best model weights
├── final_model.pth        # Final trained model
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd traffic-sign-recognition
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python src/train_fixed.py
   ```

5. **Evaluate the model**
   ```bash
   python src/evaluate.py
   ```

## 📊 Results

After training, you'll find:
- Training/validation loss and accuracy plots
- Confusion matrix saved as `confusion_matrix.png`
- Detailed classification report in the console

## 📝 Notes

- The model was trained for 5 epochs with early stopping
- Data augmentation includes random rotations, flips, and color jitter
- The best model is saved as `best_model.pth` based on validation accuracy

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
