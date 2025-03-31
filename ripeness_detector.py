import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os

class RipenessDetector:
    def __init__(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load models
        self.detection_model = self.load_detection_model()
        self.classification_models = self.load_classification_models()
        
        # Define ripeness classes
        self.classes = ['Unripe', 'Ripe', 'Overripe/Spoiled']
        
        # Define supported fruits
        self.supported_fruits = ['banana', 'orange', 'pineapple']
        
    def load_detection_model(self):
        # Load the object detection model
        model_path = "Detection models/inference_graph1/saved_model"
        if not os.path.exists(model_path):
            raise Exception(f"Detection model not found at {model_path}")
        return tf.saved_model.load(model_path)
    
    def load_classification_models(self):
        # Load the ripeness classification models
        models = {}
        model_dir = "Classification models"
        if not os.path.exists(model_dir):
            raise Exception(f"Classification models directory not found at {model_dir}")
            
        for fruit in self.supported_fruits:
            model_path = os.path.join(model_dir, f"{fruit}.pbz2")
            if os.path.exists(model_path):
                models[fruit] = tf.saved_model.load(model_path)
            else:
                print(f"Warning: Model for {fruit} not found at {model_path}")
        
        return models
    
    def preprocess_image(self, image):
        # Resize image to match model input size
        image = cv2.resize(image, (224, 224))
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect_fruits(self, frame):
        # Convert frame to tensor
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run detection
        detections = self.detection_model(input_tensor)
        
        # Process detections
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        
        # Filter detections with confidence > 0.5
        valid_detections = scores > 0.5
        
        return boxes[valid_detections], scores[valid_detections], classes[valid_detections]
    
    def detect_ripeness(self, frame, fruit_type):
        if fruit_type not in self.classification_models:
            return None, 0.0
            
        # Preprocess the frame
        processed_frame = self.preprocess_image(frame)
        
        # Get prediction from the appropriate model
        model = self.classification_models[fruit_type]
        predictions = model(processed_frame)
        
        # Get the predicted class and confidence
        prediction = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.classes[prediction], confidence
    
    def draw_detections(self, frame, boxes, scores, classes):
        height, width = frame.shape[:2]
        
        for box, score, class_id in zip(boxes, scores, classes):
            # Convert box coordinates to pixel coordinates
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Get fruit type and ripeness
            fruit_type = self.supported_fruits[int(class_id) - 1]
            ripeness, confidence = self.detect_ripeness(frame[ymin:ymax, xmin:xmax], fruit_type)
            
            if ripeness:
                # Draw label
                label = f"{fruit_type}: {ripeness} ({confidence:.2f})"
                cv2.putText(frame, label, (xmin, ymin - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def run(self):
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Detect fruits
                boxes, scores, classes = self.detect_fruits(frame)
                
                # Draw detections and ripeness information
                self.draw_detections(frame, boxes, scores, classes)
                
                # Display the frame
                cv2.imshow('Fruit Ripeness Detection', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = RipenessDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}") 