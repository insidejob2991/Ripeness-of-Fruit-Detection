import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

class RipenessDetector:
    def __init__(self, model_path='ripeness_model.h5', class_indices_path='class_indices.json'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping for class indices
        self.class_names = {v: k for k, v in self.class_indices.items()}
        
        # Define conditions
        self.conditions = ['Fresh', 'Rotten', 'Formalin-mixed']
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def preprocess_image(self, image):
        # Resize image
        image = cv2.resize(image, (224, 224))
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect(self, frame):
        # Preprocess the frame
        processed_frame = self.preprocess_image(frame)
        
        # Get prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get fruit name
        fruit = self.class_names[predicted_class]
        
        # Simple condition detection based on color and texture
        # Convert frame to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate average saturation and value
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Simple condition classification based on color characteristics
        if avg_value < 100:  # Dark image
            condition = 'Rotten'
        elif avg_saturation < 50:  # Low saturation
            condition = 'Formalin-mixed'
        else:
            condition = 'Fresh'
        
        return fruit, condition, confidence
    
    def run(self):
        print("\nPress 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect ripeness
            fruit, condition, confidence = self.detect(frame)
            
            # Draw results on frame
            # Draw background rectangle
            cv2.rectangle(frame, (5, 5), (400, 100), (0, 0, 0), -1)
            
            # Draw text with white color
            cv2.putText(frame, f"Fruit: {fruit}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Condition: {condition}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Fruit Ripeness Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = RipenessDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}") 