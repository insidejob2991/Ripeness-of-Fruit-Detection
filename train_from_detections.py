import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import os

class DetectionTrainer:
    def __init__(self, model_path='ripeness_model.h5', class_indices_path='class_indices.json'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping for class indices
        self.class_names = {v: k for k, v in self.class_indices.items()}
        
        # Create training data directory
        self.training_dir = Path('live_training_data')
        self.training_dir.mkdir(exist_ok=True)
        
        # Load existing training data if any
        self.training_data = []
        self.load_training_data()
    
    def load_training_data(self):
        """Load existing training data from disk"""
        training_file = self.training_dir / 'training_data.json'
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"\nLoaded {len(self.training_data)} training examples")
    
    def save_training_data(self):
        """Save training data to disk"""
        training_file = self.training_dir / 'training_data.json'
        with open(training_file, 'w') as f:
            json.dump(self.training_data, f)
        print(f"\nSaved {len(self.training_data)} training examples")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
        
        # Resize image
        image = cv2.resize(image, (224, 224))
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def show_class_options(self):
        """Display available class options"""
        print("\nAvailable classes:")
        print("1: Apple Fresh")
        print("2: Apple Rotten")
        print("3: Apple Formalin-mixed")
        print("4: Banana Fresh")
        print("5: Banana Rotten")
        print("6: Banana Formalin-mixed")
        print("7: Orange Fresh")
        print("8: Orange Rotten")
        print("9: Orange Formalin-mixed")
    
    def process_detection(self, image_path, text_path):
        """Process a single detection image and its text file"""
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Read the text file
        with open(text_path, 'r') as f:
            lines = f.readlines()
        
        # Display the image
        cv2.imshow('Detection', image)
        
        # Show the current classification
        print("\nCurrent classification:")
        for line in lines[1:]:  # Skip the header line
            print(line.strip())
        
        # Define class mapping that matches class_indices.json
        class_mapping = {
            1: "Apple",
            2: "Banana",
            3: "Orange",
            4: "Mango"
        }
        
        condition_mapping = {
            'a': "Fresh",
            'b': "Rotten",
            'c': "Formalin-mixed"
        }
        
        # Display options
        print("\nSelect fruit type:")
        for num, name in class_mapping.items():
            print(f"{num}: {name}")
        
        print("\nSelect condition:")
        for key, condition in condition_mapping.items():
            print(f"{key}: {condition}")
        
        # Ask for correction
        while True:
            try:
                fruit_choice = input("\nEnter the fruit number (or 's' to skip, 'q' to quit): ")
                if fruit_choice.lower() == 'q':
                    return False
                if fruit_choice.lower() == 's':
                    return True
                
                fruit_num = int(fruit_choice)
                if 1 <= fruit_num <= 4:
                    # Get fruit name
                    fruit_name = class_mapping[fruit_num]
                    
                    # Get condition
                    while True:
                        condition_choice = input("Enter condition (a/b/c): ").lower()
                        if condition_choice in condition_mapping:
                            condition = condition_mapping[condition_choice]
                            break
                        print("Invalid condition. Please enter a, b, or c.")
                    
                    # Save the training example
                    timestamp = image_path.stem.split('_')[1]  # Get timestamp from filename
                    training_image_path = self.training_dir / f"training_{timestamp}.jpg"
                    cv2.imwrite(str(training_image_path), image)
                    
                    # Add to training data
                    self.training_data.append({
                        'image_path': str(training_image_path),
                        'class': fruit_name,
                        'condition': condition,
                        'timestamp': timestamp
                    })
                    
                    # Save updated training data
                    self.save_training_data()
                    
                    print(f"\nAdded training example for: {fruit_name} ({condition})")
                    return True
                else:
                    print("Invalid fruit number. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a number or 's'/'q'.")
    
    def retrain_model(self):
        """Retrain the model with the updated training data"""
        if len(self.training_data) < 5:
            print("\nNeed at least 5 training examples to retrain the model")
            return
        
        print("\nRetraining model with new data...")
        
        # Prepare training data
        X = []
        y = []
        
        for example in self.training_data:
            # Load and preprocess image
            image = cv2.imread(example['image_path'])
            if image is not None:
                processed_image = self.preprocess_image(image)
                X.append(processed_image[0])  # Remove batch dimension
                try:
                    # Get class index for the base fruit type
                    class_index = self.class_indices[example['class']]
                    y.append(class_index)
                    print(f"Processing example: {example['class']} ({example['condition']}) -> class index {class_index}")
                except KeyError:
                    print(f"\nWarning: Class '{example['class']}' not found in class indices.")
                    print("Available classes:", list(self.class_indices.keys()))
                    continue
        
        if not X or not y:
            print("\nNo valid training examples found!")
            return
        
        if len(X) != len(y):
            print(f"\nError: Mismatch in data sizes - X: {len(X)}, y: {len(y)}")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTraining with {len(X)} examples")
        
        # Fine-tune the model
        self.model.fit(
            X, y,
            epochs=5,
            batch_size=min(32, len(X)),
            validation_split=0.2,
            verbose=1
        )
        
        # Save the retrained model
        self.model.save('ripeness_model.h5')
        print("\nModel retraining complete and saved!")
    
    def clear_detected_fruits(self):
        """Clear all files from the detected_fruits folder"""
        detected_dir = Path('detected_fruits')
        if detected_dir.exists():
            for file in detected_dir.glob('*'):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Could not delete {file}: {e}")
            print("\nCleared detected_fruits folder")
    
    def run(self):
        """Process all detections in the detected_fruits folder"""
        # Clear previous detections
        self.clear_detected_fruits()
        
        detected_dir = Path('detected_fruits')
        if not detected_dir.exists():
            print("detected_fruits folder not found!")
            return
        
        # Get all detection images
        image_files = list(detected_dir.glob('detection_*.jpg'))
        if not image_files:
            print("No detection images found!")
            return
        
        print(f"\nFound {len(image_files)} detection images")
        
        # Process each detection
        for image_path in image_files:
            text_path = image_path.with_suffix('.txt')
            if not text_path.exists():
                print(f"No text file found for {image_path}")
                continue
            
            print(f"\nProcessing {image_path.name}")
            if not self.process_detection(image_path, text_path):
                break
        
        # Ask if user wants to retrain the model
        if len(self.training_data) >= 5:
            choice = input("\nDo you want to retrain the model with the new examples? (y/n): ")
            if choice.lower() == 'y':
                self.retrain_model()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        trainer = DetectionTrainer()
        trainer.run()
    except Exception as e:
        print(f"Error: {str(e)}") 