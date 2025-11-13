import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch, 
    ImageFileCreateEntry,
    ProjectSettings
)
from msrest.authentication import ApiKeyCredentials

def upload_images(trainer, project, dataset_path, tag_name, tag_id=None):
    """Upload images to Custom Vision project"""
    if tag_id is None:
        # Create a new tag if not provided
        tag = trainer.create_tag(project.id, tag_name)
        tag_id = tag.id
    
    image_dir = Path(dataset_path) / tag_name
    if not image_dir.exists():
        print(f"Warning: Directory {image_dir} does not exist")
        return tag_id, 0
    
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    print(f"Uploading {len(image_files)} images for tag '{tag_name}'...")
    
    # Upload in batches of 64 (API limit)
    for i in range(0, len(image_files), 64):
        batch = image_files[i:i + 64]
        image_entries = []
        
        for img_path in batch:
            with open(img_path, 'rb') as img_data:
                image_entries.append(ImageFileCreateEntry(
                    name=img_path.name,
                    contents=img_data.read(),
                    tag_ids=[tag_id]
                ))
        
        upload_result = trainer.create_images_from_files(
            project.id,
            ImageFileCreateBatch(images=image_entries)
        )
        
        if not upload_result.is_batch_successful:
            print("Some images failed to upload:")
            for image in upload_result.images:
                if image.status != "OK":
                    print(f"  {image.source_url}: {image.status}")
    
    return tag_id, len(image_files)

def train_model(trainer, project, model_name="TrafficSignModel"):
    """Train the model and wait for completion"""
    print("Training model...")
    iteration = trainer.train_project(project.id)
    
    # Wait for training to complete
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        print(f"Training status: {iteration.status}")
        time.sleep(10)
    
    print("Training complete!")
    
    # Update iteration with a name
    iteration = trainer.update_iteration(
        project.id,
        iteration.id,
        name=model_name,
        is_default=True
    )
    
    return iteration

def publish_model(trainer, project, iteration, prediction_resource_id):
    """Publish the trained model"""
    publish_iteration_name = "traffic-sign-classifier"
    
    # The prediction resource ID is the same as the training resource ID
    # in the same region
    print("Publishing model...")
    trainer.publish_iteration(
        project.id,
        iteration.id,
        publish_iteration_name,
        prediction_resource_id
    )
    
    print(f"Model published as '{publish_iteration_name}'")
    return publish_iteration_name

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a custom vision model for traffic sign recognition')
    parser.add_argument('--dataset_path', type=str, default='../data/processed',
                       help='Path to the processed dataset directory (should contain train/val/test subdirectories)')
    parser.add_argument('--project_name', type=str, default='TrafficSignRecognition',
                       help='Name of the Custom Vision project')
    parser.add_argument('--model_name', type=str, default='TrafficSignModel',
                       help='Name to give to the trained model')
    
    args = parser.parse_args()
    
    # Azure Custom Vision credentials
    training_key = os.getenv('CUSTOM_VISION_TRAINING_KEY')
    endpoint = os.getenv('CUSTOM_VISION_ENDPOINT')
    prediction_key = os.getenv('CUSTOM_VISION_PREDICTION_KEY')
    
    if not all([training_key, endpoint, prediction_key]):
        raise ValueError("Please set all required environment variables in .env file")
    
    # Authenticate the client
    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(endpoint, credentials)
    
    print(f"Creating project '{args.project_name}'...")
    
    # Check if project already exists
    projects = trainer.get_projects()
    project = next((p for p in projects if p.name == args.project_name), None)
    
    if project is None:
        # Create a new project
        project = trainer.create_project(
            name=args.project_name,
            classification_type="Multiclass",
            domain_id="0732100f-1a38-4e49-a514-c9b0c035b7f6"  # General (compact) domain
        )
        print(f"Created new project with ID: {project.id}")
    else:
        print(f"Using existing project with ID: {project.id}")
    
    # Upload training images
    train_path = Path(args.dataset_path) / 'train'
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found at {train_path}")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(train_path) 
                 if (train_path / d).is_dir()]
    
    print(f"Found {len(class_dirs)} classes in the dataset")
    
    # Upload images for each class
    tag_map = {}
    for class_name in tqdm(class_dirs, desc="Uploading training images"):
        tag_id, count = upload_images(trainer, project, train_path, class_name)
        if count > 0:
            tag_map[class_name] = tag_id
    
    if not tag_map:
        raise ValueError("No training images were uploaded. Check your dataset path.")
    
    # Train the model
    iteration = train_model(trainer, project, args.model_name)
    
    # Publish the model
    # The prediction resource ID is typically the same as the training resource ID
    # in the same region. You can find this in the Azure Portal under the resource's
    # properties as 'Resource ID'.
    prediction_resource_id = input("\nPlease enter your Prediction Resource ID from Azure Portal: ")
    
    if prediction_resource_id.strip():
        model_name = publish_model(trainer, project, iteration, prediction_resource_id)
        print(f"\nModel training and publishing complete!")
        print(f"Project ID: {project.id}")
        print(f"Iteration ID: {iteration.id}")
        print(f"Published as: {model_name}")
    else:
        print("\nSkipping model publishing. The model is trained but not published.")
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
