# Fruit and Vegetable Ripeness Detector

A comprehensive system for detecting and monitoring the ripeness of fruits and vegetables using computer vision, deep learning, and sensor data. This application provides both real-time monitoring and a web interface for fruit ripeness detection.

## Features

- Real-time camera feed processing with ripeness classification
- Web-based user interface with authentication
- Sensor monitoring and calibration system
- Fruit detection and classification
- Ripeness status tracking (Unripe, Ripe, Overripe/Spoiled)
- Confidence score display
- Data logging and analysis
- User management system
- Image capture and storage
- Training data collection and model training capabilities

## System Components

- **Web Interface**: Flask-based web application for user interaction
- **Detection System**: YOLO-based fruit detection ([YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5))
- **Classification System**: Deep learning model for ripeness classification
- **Sensor Monitoring**: Real-time sensor data collection and analysis
- **Data Management**: Organized dataset structure and training pipeline

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt):
  - Flask 2.0.1
  - Flask-SocketIO 5.1.1
  - Flask-Login 0.5.0
  - OpenCV 4.5.3.56
  - NumPy 1.21.2
  - TensorFlow 2.6.0
  - And other dependencies

## Installation

1. Clone this repository
2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Download YOLO files (if needed):
```bash
python download_yolo.py
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Access the web interface through your browser (default: http://localhost:5000)

3. For sensor calibration:
```bash
-- calibrate in clean air 
python calibrate_sensors.py
```

4. To monitor sensors:
```bash
-- in the average value of sensors in monitor.py update the calibration values 
python monitor_sensors.py
```

5. For training the model:
```bash
# Main training script for initial model training
python train_ripeness_model.py
-- put the photos of fruit in the filesystem as its given ie apple-rotten,fresh etc.

# Optional: Retrain model with new detections
python train_from_detections.py
-- This script works with already detected fruits in the detected_fruits directory
-- It's useful for fine-tuning the model with new data
```

## Project Structure

- `app.py`: Main web application
- `detect_fruits.py`: Fruit detection implementation
- `train_ripeness_model.py`: Model training script
- `monitor_sensors.py`: Sensor monitoring system
- `calibrate_sensors.py`: Sensor calibration utility
- `templates/`: Web interface templates
- `static/`: Static web assets
- `dataset/`: Training and testing data
- `detected_fruits/`: Storage for detected fruit images
- `models/`: Trained model files

## Note

This is a production-ready implementation that includes:
- User authentication and management
- Real-time sensor monitoring
- Comprehensive data collection and training pipeline
- Web-based interface for easy interaction
- Robust error handling and logging

## Future Improvements

- Add support for more fruit/vegetable types
- Implement mobile application
- Add API endpoints for external system integration
- Enhance data visualization and reporting
- Implement automated alerts for ripeness changes
- Add batch processing capabilities 
- Add better hardeware for more accurate gas detection
- Add microphone module to detect sound of rodents, worms etc.
- Add automatic food addition and deletion via barcode scanning 
- Add ir camera for better and effective visuals