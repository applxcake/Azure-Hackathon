import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image
import matplotlib.pyplot as plt

def predict_image(predictor, project_id, publish_iteration_name, image_path, threshold=0.5):
    """Make a prediction on a single image"""
    with open(image_path, 'rb') as image_data:
        results = predictor.classify_image(
            project_id,
            publish_iteration_name,
            image_data.read()
        )
    
    # Filter predictions above threshold
    predictions = [
        (pred.tag_name, pred.probability)
        for pred in results.predictions
        if pred.probability >= threshold
    ]
    
    # Sort by probability in descending order
    return sorted(predictions, key=lambda x: x[1], reverse=True)

def display_prediction(image_path, predictions):
    """Display the image with prediction results"""
    img = Image.open(image_path)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    
    # Create prediction text
    pred_text = "Predictions:\n"
    for i, (tag, prob) in enumerate(predictions[:5]):  # Show top 5 predictions
        pred_text += f"{i+1}. {tag}: {prob*100:.2f}%\n"
    
    plt.figtext(0.5, 0.01, pred_text, 
                ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using the trained model')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the image file for prediction')
    parser.add_argument('--project_id', type=str, required=True,
                       help='Project ID from Custom Vision')
    parser.add_argument('--publish_iteration_name', type=str, 
                       default='traffic-sign-classifier',
                       help='Published iteration name (default: traffic-sign-classifier)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    
    args = parser.parse_args()
    
    # Azure Custom Vision credentials
    prediction_key = os.getenv('CUSTOM_VISION_PREDICTION_KEY')
    endpoint = os.getenv('CUSTOM_VISION_ENDPOINT')
    
    if not all([prediction_key, endpoint]):
        raise ValueError("Please set CUSTOM_VISION_PREDICTION_KEY and CUSTOM_VISION_ENDPOINT in .env file")
    
    # Create prediction client
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint, credentials)
    
    # Make prediction
    predictions = predict_image(
        predictor=predictor,
        project_id=args.project_id,
        publish_iteration_name=args.publish_iteration_name,
        image_path=args.image_path,
        threshold=args.threshold
    )
    
    # Display results
    if predictions:
        print("\nPredictions:")
        for tag, prob in predictions:
            print(f"- {tag}: {prob*100:.2f}%")
        
        # Display image with predictions
        try:
            display_prediction(args.image_path, predictions)
        except Exception as e:
            print(f"\nNote: Could not display image. {str(e)}")
    else:
        print("No predictions met the confidence threshold.")

if __name__ == "__main__":
    main()
