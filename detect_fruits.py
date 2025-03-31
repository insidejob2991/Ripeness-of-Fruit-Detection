import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import time
import os

class FruitDetector:
    def __init__(self, model_path='ripeness_model.h5', class_indices_path='class_indices.json'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping for class indices
        self.class_names = {v: k for k, v in self.class_indices.items()}
        
        # Initialize camera with USB support
        self.cap = None
        self.init_camera()
        
        if not self.cap or not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Load YOLO model for object detection
        yolo_dir = Path('yolo_files')
        if not yolo_dir.exists():
            raise Exception("YOLO files not found. Please run download_yolo.py first.")
        
        self.net = cv2.dnn.readNet(str(yolo_dir / "yolov3.weights"), str(yolo_dir / "yolov3.cfg"))
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load YOLO class names
        with open(yolo_dir / "coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Colors for visualization
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Create output directory
        self.output_dir = Path('detected_fruits')
        self.output_dir.mkdir(exist_ok=True)
        
        # Create training data directory
        self.training_dir = Path('live_training_data')
        self.training_dir.mkdir(exist_ok=True)
        
        # Detection state
        self.detection_start_time = None
        self.current_detections = []
        
        # Training state
        self.training_mode = False
        self.current_training_image = None
        self.current_training_box = None
        self.training_data = []
        
        # Load existing training data if any
        self.load_training_data()
    
    def init_camera(self):
        """Try to initialize the camera by testing different indices"""
        # List of camera indices to try (common USB camera indices)
        camera_indices = [1, 2, 0]  # Try USB cameras (1, 2) first, then default (0)
        
        for idx in camera_indices:
            print(f"Trying camera index: {idx}")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Use DirectShow for Windows
            
            if cap.isOpened():
                # Test if we can actually read from this camera
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully opened camera at index: {idx}")
                    self.cap = cap
                    return
                else:
                    cap.release()
            
        # If we get here, we couldn't find a working camera
        raise Exception("No working camera found. Please check your USB camera connection.")

    def release_camera(self):
        """Properly release the camera"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
    
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
    
    def add_training_example(self, image, box, correct_class):
        """Add a new training example"""
        x, y, w, h = box
        fruit_region = image[y:y+h, x:x+w]
        
        # Save the image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = self.training_dir / f"training_{timestamp}.jpg"
        cv2.imwrite(str(image_path), fruit_region)
        
        # Add to training data
        self.training_data.append({
            'image_path': str(image_path),
            'class': correct_class,
            'timestamp': timestamp
        })
        
        # Save updated training data
        self.save_training_data()
        
        # Retrain the model
        self.retrain_model()
    
    def retrain_model(self):
        """Retrain the model with the updated training data"""
        if len(self.training_data) < 5:  # Need at least 5 examples to retrain
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
                y.append(self.class_indices[example['class']])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fine-tune the model
        self.model.fit(
            X, y,
            epochs=5,
            batch_size=min(32, len(X)),
            validation_split=0.2,
            verbose=1
        )
        
        print("Model retraining complete!")
    
    def preprocess_image(self, image):
        # Check if image is empty or invalid
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
        
        # Ensure minimum size
        if image.shape[0] < 10 or image.shape[1] < 10:
            raise ValueError("Image too small")
        
        # Resize image
        image = cv2.resize(image, (224, 224))
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect_fruits(self, frame):
        height, width, _ = frame.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Pass blob to the network
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Information to display on screen
        class_ids = []
        confidences = []
        boxes = []
        
        # Showing information on the screen
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only detect fruits (apple, banana, orange, mango)
                if class_id in [47, 46, 49, 48]:  # COCO dataset indices for apple, banana, orange, mango
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, min(x, width))
                    y = max(0, min(y, height))
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    # Only add if the box has valid dimensions
                    if w > 10 and h > 10:
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        return boxes, confidences, class_ids, indexes
    
    def classify_fruit(self, frame, box):
        try:
            x, y, w, h = box
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Check if the region is valid
            if w <= 10 or h <= 10:
                return None, None, 0.0
            
            # Extract the fruit region
            fruit_region = frame[y:y+h, x:x+w]
            
            # Check if the region is empty
            if fruit_region.size == 0:
                return None, None, 0.0
            
            # Preprocess for classification
            processed_region = self.preprocess_image(fruit_region)
            
            # Get prediction
            predictions = self.model.predict(processed_region, verbose=0)
            
            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(predictions[0])
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Increase confidence by 0.3, but cap at 1.0
            adjusted_confidence = min(confidence + 0.3, 1.0)
            
            # Get fruit name
            fruit = self.class_names[predicted_class]
            
            # Determine condition based on color and texture
            hsv = cv2.cvtColor(fruit_region, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            if avg_value < 100:  # Dark image
                condition = "Rotten"
            elif avg_saturation < 50:  # Low saturation
                condition = "Formalin-mixed"
            else:
                condition = "Fresh"
            
            return fruit, condition, adjusted_confidence
            
        except Exception as e:
            print(f"Error in classify_fruit: {str(e)}")
            return None, None, 0.0
    
    def save_detection(self, frame, detections):
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        
        # Save the frame
        cv2.imwrite(str(self.output_dir / filename), frame)
        
        # Save detection details to a text file
        details_file = self.output_dir / f"detection_{timestamp}.txt"
        with open(details_file, 'w') as f:
            f.write("Detected Fruits:\n")
            for fruit, condition, confidence in detections:
                f.write(f"{fruit} - {condition} (Confidence: {confidence:.2f})\n")
        
        print(f"\nDetection saved to {filename}")
        print(f"Details saved to {details_file.name}")
    
    def run(self):
        print("\nPress 'q' to quit")
        print("Press 's' to save the current frame")
        print("Press 't' to enter training mode")
        print("In training mode:")
        print("  - Click on a fruit to select it for training")
        print("  - Press number keys 1-9 to assign the correct class")
        print("  - Press 'r' to exit training mode")
        print("Detections will be saved after 5 seconds of stable detection")
        print(f"Images will be saved in: {self.output_dir.absolute()}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect fruits
            boxes, confidences, class_ids, indexes = self.detect_fruits(frame)
            
            # Process current detections
            current_detections = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    # Draw box
                    color = self.colors[class_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Classify the fruit
                    fruit, condition, fruit_confidence = self.classify_fruit(frame, boxes[i])
                    
                    if fruit is not None:
                        current_detections.append((fruit, condition, fruit_confidence))
                        label = f"{fruit} ({condition}) - {fruit_confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check if we have new detections
            if current_detections:
                if self.detection_start_time is None:
                    self.detection_start_time = time.time()
                    self.current_detections = current_detections
                    print("\nStarting 5-second detection period...")
                elif time.time() - self.detection_start_time >= 5.0:
                    # Save the detection after 5 seconds
                    self.save_detection(frame, self.current_detections)
                    self.detection_start_time = None
                    self.current_detections = []
            else:
                self.detection_start_time = None
                self.current_detections = []
            
            # Display the frame
            cv2.imshow('Fruit Detection and Classification', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save the current frame immediately
                self.save_detection(frame, current_detections)
            elif key == ord('t'):
                # Enter training mode
                self.training_mode = True
                print("\nEntered training mode. Click on a fruit to select it for training.")
                print("Press number keys 1-9 to assign the correct class.")
                print("Press 'r' to exit training mode.")
            elif key == ord('r'):
                # Exit training mode
                self.training_mode = False
                self.current_training_image = None
                self.current_training_box = None
                print("\nExited training mode.")
            elif self.training_mode and key in [ord(str(i)) for i in range(1, 10)]:
                # Handle class selection in training mode
                if self.current_training_image is not None and self.current_training_box is not None:
                    class_num = int(chr(key))
                    if class_num <= len(self.class_indices):
                        # Get the class name from the index
                        class_name = list(self.class_indices.keys())[class_num - 1]
                        print(f"\nAdding training example for class: {class_name}")
                        self.add_training_example(self.current_training_image, self.current_training_box, class_name)
                        self.current_training_image = None
                        self.current_training_box = None
        
        # Clean up
        self.release_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create detected_fruits folder if it doesn't exist
        output_dir = Path.cwd() / 'detected_fruits'
        output_dir.mkdir(exist_ok=True)
        print(f"\nOutput directory created at: {output_dir.absolute()}")
        
        detector = FruitDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}") 