import cv2
import os
from pathlib import Path
import time

class ImageCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Define fruits and conditions
        self.fruits = ['Apple', 'Banana', 'Orange']
        self.conditions = ['Fresh', 'Rotten', 'Formalin-mixed']
        
        # Create dataset structure if it doesn't exist
        self.create_dataset_structure()
    
    def create_dataset_structure(self):
        dataset_dir = Path('dataset')
        dataset_dir.mkdir(exist_ok=True)
        
        for fruit in self.fruits:
            fruit_dir = dataset_dir / fruit
            fruit_dir.mkdir(exist_ok=True)
            
            for condition in self.conditions:
                condition_dir = fruit_dir / condition
                condition_dir.mkdir(exist_ok=True)
    
    def capture_images(self):
        current_fruit = 0
        current_condition = 0
        image_count = 0
        
        print("\nInstructions:")
        print("1. Use arrow keys to navigate:")
        print("   - Up/Down: Change fruit type")
        print("   - Left/Right: Change condition")
        print("2. Press SPACE to capture an image")
        print("3. Press ESC to exit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display current settings
            fruit = self.fruits[current_fruit]
            condition = self.conditions[current_condition]
            cv2.putText(frame, f"Fruit: {fruit}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Condition: {condition}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images captured: {image_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Capture Training Images', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                # Save image
                image_path = f"dataset/{fruit}/{condition}/image_{image_count}.jpg"
                cv2.imwrite(image_path, frame)
                image_count += 1
                print(f"Saved image: {image_path}")
            elif key == 82:  # Up arrow
                current_fruit = (current_fruit + 1) % len(self.fruits)
            elif key == 84:  # Down arrow
                current_fruit = (current_fruit - 1) % len(self.fruits)
            elif key == 81:  # Left arrow
                current_condition = (current_condition - 1) % len(self.conditions)
            elif key == 83:  # Right arrow
                current_condition = (current_condition + 1) % len(self.conditions)
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        capture = ImageCapture()
        capture.capture_images()
    except Exception as e:
        print(f"Error: {str(e)}") 