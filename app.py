from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import time
import os
from detect_fruits import FruitDetector
import threading
import logging
from collections import defaultdict
from monitor_sensors import SensorMonitor
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow to reduce warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure Socket.IO with improved settings
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=300,  # 5 minutes
    ping_interval=25,
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    max_http_buffer_size=1e8  # Increase buffer size for video frames
)

# User model
class User(UserMixin):
    def __init__(self, id, email, password_hash):
        self.id = id
        self.email = email
        self.password_hash = password_hash
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'password_hash': self.password_hash
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['id'], data['email'], data['password_hash'])

# File path for storing users
USERS_FILE = 'users.json'

def load_users():
    """Load users from JSON file"""
    try:
        if Path(USERS_FILE).exists():
            with open(USERS_FILE, 'r') as f:
                users_data = json.load(f)
                return {email: User.from_dict(data) for email, data in users_data.items()}
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
    return {}

def save_users():
    """Save users to JSON file"""
    try:
        users_data = {email: user.to_dict() for email, user in users.items()}
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")

# Load users at startup
users = load_users()

# If no users exist, create default admin user
if not users:
    users = {
        'admin@example.com': User(1, 'admin@example.com', generate_password_hash('admin123'))
    }
    save_users()

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if str(user.id) == str(user_id):
            return user
    return None

# Global variables
detector = None
camera_thread = None
is_camera_active = False
latest_frame = None
latest_detections = []
connected_clients = set()  # Use a set to track unique client IDs
last_heartbeat = {}  # Track last heartbeat time for each client
most_recent_detection = None  # Store the most recent detection globally
current_inventory = []  # Change to a list instead of defaultdict

# Initialize sensor monitor and data
sensor_monitor = SensorMonitor()
sensor_data = {
    'temperature': 0,
    'humidity': 0,
    'mq135': 0,
    'mq2': 0,
    'door_status': 'closed',
    'power_status': 'on',
    'last_update': None
}

def init_detector():
    global detector
    try:
        detector = FruitDetector()
        logger.info("Successfully initialized detector and camera")
        return True
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error initializing detector: {error_msg}")
        if "No working camera found" in error_msg:
            socketio.emit('camera_error', 
                         {'message': 'USB camera not found. Please check the connection.'},
                         room='viewers')
        elif "Could not open camera" in error_msg:
            socketio.emit('camera_error', 
                         {'message': 'Failed to access camera. Please check permissions.'},
                         room='viewers')
        else:
            socketio.emit('camera_error', 
                         {'message': f'Camera initialization error: {error_msg}'},
                         room='viewers')
        return False

@socketio.on('heartbeat')
def handle_heartbeat():
    """Handle client heartbeat to keep connection alive"""
    client_id = request.sid
    last_heartbeat[client_id] = time.time()
    return {'status': 'alive'}

def camera_stream():
    global latest_frame, latest_detections, is_camera_active, detector
    detection_start_time = None  # Track when stable detection started
    last_detections = None  # Track last stable detections
    last_save_time = None  # Track when we last saved
    output_dir = Path('detected_fruits')
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Camera stream started. Saving detections to: {output_dir.absolute()}")

    reconnect_delay = 1  # Initial reconnect delay in seconds
    max_reconnect_delay = 30  # Maximum reconnect delay in seconds

    # Add heartbeat check
    last_heartbeat_check = time.time()
    heartbeat_interval = 30  # Check every 30 seconds

    def save_detection(frame, detections, timestamp):
        try:
            # Create paths
            image_path = output_dir / f"detection_{timestamp}.jpg"
            text_path = output_dir / f"detection_{timestamp}.txt"
            
            # Save the image
            success = cv2.imwrite(str(image_path), frame)
            if not success:
                raise Exception("Failed to save image")
            
            # Save detection details
            with open(text_path, 'w') as f:
                f.write("Detected Fruits:\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("-" * 30 + "\n")
                for det in detections:
                    f.write(f"Fruit: {det['fruit']}\n")
                    f.write(f"Condition: {det['condition']}\n")
                    f.write(f"Confidence: {det['confidence']:.2f}\n")
                    f.write("-" * 30 + "\n")
            
            logger.info(f"Successfully saved detection to {image_path}")
            
            # Get complete detection info and emit to clients
            recent_detection = get_most_recent_detection()
            socketio.emit('recent_detection_update', recent_detection, room='viewers')
            
            return str(image_path)
        except Exception as e:
            logger.error(f"Error saving detection: {str(e)}")
            return None

    while is_camera_active:
        try:
            current_time = time.time()
            
            # Check client heartbeats periodically
            if current_time - last_heartbeat_check >= heartbeat_interval:
                for client_id in list(connected_clients):
                    if client_id in last_heartbeat:
                        if current_time - last_heartbeat[client_id] > 60:  # No heartbeat for 1 minute
                            logger.warning(f"Client {client_id} heartbeat timeout")
                            socketio.emit('reconnect_required', room=client_id)
                last_heartbeat_check = current_time

            # Check if camera is opened, if not try to reinitialize
            if not detector or not detector.cap or not detector.cap.isOpened():
                logger.warning("Camera disconnected, attempting to reconnect...")
                socketio.emit('camera_error', 
                            {'message': 'Camera disconnected, attempting to reconnect...'},
                            room='viewers')
                
                # Release existing camera if any
                if detector:
                    detector.release_camera()
                
                # Reinitialize detector
                if not init_detector():
                    logger.error(f"Failed to reconnect, retrying in {reconnect_delay} seconds...")
                    time.sleep(reconnect_delay)
                    # Increase reconnect delay with exponential backoff
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue
                else:
                    logger.info("Successfully reconnected to camera")
                    socketio.emit('camera_status', {'message': 'Camera reconnected'}, room='viewers')
                    reconnect_delay = 1  # Reset reconnect delay
            
            ret, frame = detector.cap.read()
            if not ret or frame is None:
                raise Exception("Failed to grab frame")
            
            # Reset reconnect delay on successful frame grab
            reconnect_delay = 1
            
            # Detect fruits
            boxes, confidences, class_ids, indexes = detector.detect_fruits(frame)
            
            # Process current detections
            current_detections = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    # Draw box
                    color = detector.colors[class_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Classify the fruit
                    fruit, condition, fruit_confidence = detector.classify_fruit(frame, boxes[i])
                    
                    if fruit is not None:
                        current_detections.append({
                            'fruit': fruit,
                            'condition': condition,
                            'confidence': float(fruit_confidence)
                        })
                        label = f"{fruit} ({condition}) - {fruit_confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Remove automatic inventory update
            # update_inventory(current_detections)
            
            # Check for stable detections
            if current_detections:  # If there are any detections
                # Check if it's time to save again (every 5 seconds)
                if last_save_time is None or (current_time - last_save_time) >= 5:
                    # Save the detection
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    saved_path = save_detection(frame, current_detections, timestamp)
                    
                    if saved_path:
                        socketio.emit('detection_saved', {
                            'image': saved_path,
                            'detections': current_detections
                        }, room='viewers')
                    
                    # Update last save time
                    last_save_time = current_time
            else:  # No detections
                last_save_time = None
            
            latest_detections = current_detections
            latest_frame = frame
            
            # Emit detections through Socket.IO
            socketio.emit('detections', current_detections, room='viewers')
            
        except Exception as e:
            logger.error(f"Error in camera stream: {str(e)}")
            socketio.emit('camera_error', {'message': str(e)}, room='viewers')
            time.sleep(1)  # Prevent rapid error loops

def generate_frames():
    while True:
        if latest_frame is not None and is_camera_active:  # Only generate frames if camera is active
            try:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error generating frame: {str(e)}")
        time.sleep(0.033)  # ~30 FPS

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return render_template('login.html', error_message='Please enter both email and password')
        
        # Check if email exists in users dictionary
        if email in users:
            # If email exists, check password
            if check_password_hash(users[email].password_hash, password):
                # Password is correct, log the user in
                user = users[email]
                login_user(user)
                return redirect(url_for('index'))
            else:
                # Email exists but password is wrong
                return render_template('login.html', error_message='Wrong password. Please try again.')
        else:
            # Email is not registered
            return render_template('login.html', error_message='Email is not registered. Please sign up first.')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if email in users:
            return render_template('signup.html', error_message='Email already registered')
        
        if password != confirm_password:
            return render_template('signup.html', error_message='Passwords do not match')
        
        if len(password) < 6:
            return render_template('signup.html', error_message='Password must be at least 6 characters long')
        
        # Create new user
        user_id = len(users) + 1
        users[email] = User(user_id, email, generate_password_hash(password))
        
        # Save users to file
        save_users()
        
        # Log the user in and redirect to home
        login_user(users[email])
        return redirect(url_for('index'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Render the home page with current sensor data"""
    return render_template('index.html', 
                         sensor_data=sensor_data if 'sensor_data' in globals() else {
                             'temperature': 0,
                             'humidity': 0,
                             'mq135': 0,
                             'mq2': 0
                         })

@app.route('/fridge_conditions')
@login_required
def fridge_conditions():
    return render_template('fridge_conditions.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('start_camera')
def handle_start_camera(data):
    global is_camera_active, camera_thread, detector
    
    if is_camera_active:
        return {'status': 'error', 'message': 'Camera already running'}
    
    try:
        if init_detector():
            is_camera_active = True
            camera_thread = threading.Thread(target=camera_stream)
            camera_thread.start()
            logger.info("Camera started successfully")
            return {'status': 'success', 'message': 'Camera started'}
        else:
            return {'status': 'error', 'message': 'Failed to initialize camera'}
    except Exception as e:
        logger.error(f"Error starting camera: {str(e)}")
        return {'status': 'error', 'message': f'Failed to start camera: {str(e)}'}

def get_most_recent_detection():
    global most_recent_detection
    if most_recent_detection is None:
        output_dir = Path('detected_fruits')
        try:
            # Get all txt files and sort by modification time
            txt_files = list(output_dir.glob('detection_*.txt'))
            if not txt_files:
                return None
            
            # Sort by modification time, newest first
            most_recent = max(txt_files, key=lambda x: x.stat().st_mtime)
            
            # Get corresponding image file
            img_file = output_dir / f"{most_recent.stem}.jpg"
            
            # Read the content
            with open(most_recent, 'r') as f:
                content = f.read()
                
            # Parse the detection information
            lines = content.strip().split('\n')
            timestamp_line = next((line for line in lines if line.startswith('Timestamp:')), None)
            timestamp = timestamp_line.replace('Timestamp:', '').strip() if timestamp_line else 'Unknown'
            
            # Parse fruit detections
            detections = []
            current_detection = {}
            for line in lines:
                if line.startswith('Fruit:'):
                    if current_detection and 'fruit' in current_detection:
                        detections.append(current_detection)
                        current_detection = {}
                    current_detection['fruit'] = line.replace('Fruit:', '').strip()
                elif line.startswith('Condition:'):
                    current_detection['condition'] = line.replace('Condition:', '').strip()
                elif line.startswith('Confidence:'):
                    current_detection['confidence'] = float(line.replace('Confidence:', '').strip())
            
            # Add the last detection if exists
            if current_detection and 'fruit' in current_detection:
                detections.append(current_detection)
            
            most_recent_detection = {
                'filename': most_recent.name,
                'image_path': str(img_file) if img_file.exists() else None,
                'content': content,
                'timestamp': timestamp,
                'detections': detections,
                'file_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', 
                                          time.localtime(most_recent.stat().st_mtime))
            }
        except Exception as e:
            logger.error(f"Error reading most recent detection: {str(e)}")
            return None
    return most_recent_detection

def update_most_recent_detection(new_detection):
    global most_recent_detection
    most_recent_detection = new_detection
    # Broadcast the update to all connected clients
    socketio.emit('recent_detection_update', new_detection, room='viewers')

@socketio.on('stop_camera')
def handle_stop_camera(data):
    global is_camera_active, camera_thread, detector
    try:
        if is_camera_active:
            logger.info("Stopping camera...")
            
            # First set the flag to stop the camera stream
            is_camera_active = False
            
            # Clear the latest frame
            latest_frame = None
            
            # Release the camera properly
            if detector:
                detector.release_camera()
                detector = None
            
            # Wait for thread to finish
            if camera_thread:
                camera_thread.join(timeout=2.0)
                camera_thread = None
            
            # Clear detections
            latest_detections.clear()
            
            # Add a small delay
            time.sleep(0.5)
            
            # Get recent detection
            recent_detection = get_most_recent_detection()
            
            logger.info("Camera stopped successfully")
            socketio.emit('camera_stopped', {
                'recent_detection': recent_detection
            }, room='viewers')
            return {'status': 'success', 'message': 'Camera stopped'}
        return {'status': 'error', 'message': 'Camera not running'}
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")
        return {'status': 'error', 'message': f'Failed to stop camera: {str(e)}'}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    global connected_clients
    client_id = request.sid
    
    # Check if this is a main client connection (not video feed)
    if 'HTTP_UPGRADE' in request.environ:
        connected_clients.add(client_id)
        logger.info(f'Main client {client_id} connected. Total main clients: {len(connected_clients)}')
        # Join a room for broadcast messages
        join_room('viewers')
        
        # Send initial fridge conditions data
        emit('temperature_update', {'temperature': 0})  # Example temperature
        emit('humidity_update', {'humidity': 0})  # Example humidity
        emit('door_status', {'isOpen': False})  # Example door status
        emit('power_status', {'isOn': True})  # Example power status

        # Send initial sensor data
        emit('sensor_update', sensor_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')
    global connected_clients
    client_id = request.sid
    if client_id in connected_clients:
        connected_clients.remove(client_id)
        leave_room('viewers')
        logger.info(f'Main client {client_id} disconnected. Total main clients: {len(connected_clients)}')

# Add a function to get accurate client count
def get_client_count():
    return len(connected_clients)

# Add a new route to get the most recent detection
@app.route('/get_recent_detection')
def get_recent_detection_route():
    detection = get_most_recent_detection()
    if detection:
        return jsonify(detection)
    else:
        return jsonify({"error": "No recent detection available"})

@socketio.on('analyze_frame')
def handle_analyze_frame(data):
    try:
        if not is_camera_active or latest_frame is None:
            return {'status': 'error', 'message': 'Camera not running or no frame available'}
        
        # Save current frame as a detection
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path('detected_fruits')
        
        # Create paths
        image_path = output_dir / f"detection_{timestamp}.jpg"
        text_path = output_dir / f"detection_{timestamp}.txt"
        
        # Save the image
        success = cv2.imwrite(str(image_path), latest_frame)
        if not success:
            raise Exception("Failed to save image")
        
        # Get current detections
        current_detections = latest_detections.copy()
        
        # Save detection details in the specified format
        with open(text_path, 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            for det in current_detections:
                f.write(f"Fruit:{det['fruit']}\n")
                f.write(f"Condition:{det['condition']}\n")
                f.write(f"Confidence:{det['confidence']:.2f}\n")
                if det != current_detections[-1]:  # If not the last detection, add a newline
                    f.write("\n")
        
        logger.info(f"Successfully saved analysis to {image_path}")
        
        # Create new detection object
        new_detection = {
            'filename': text_path.name,
            'image_path': str(image_path),
            'content': open(text_path, 'r').read(),
            'timestamp': timestamp,
            'detections': current_detections,
            'file_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update the most recent detection
        update_most_recent_detection(new_detection)
        
        return {
            'status': 'success',
            'recent_detection': new_detection,
            'num_detections': len(current_detections),
            'timestamp': timestamp,
            'image_path': str(image_path)
        }
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        return {'status': 'error', 'message': f'Failed to analyze frame: {str(e)}'}

@socketio.on('get_recent_detection')
def handle_get_recent_detection(callback):
    """Handle client request for most recent detection"""
    detection = get_most_recent_detection()
    if callback:
        callback(detection)
    return detection

# Simulate fridge condition updates
def simulate_fridge_updates():
    while True:
        if connected_clients:
            # Simulate temperature fluctuations
            temp = 4.5 + np.random.normal(0, 0.2)
            socketio.emit('temperature_update', {'temperature': round(temp, 1)})
            
            # Simulate humidity fluctuations
            humidity = 85 + np.random.normal(0, 2)
            humidity = max(0, min(100, humidity))  # Clamp between 0 and 100
            socketio.emit('humidity_update', {'humidity': round(humidity, 1)})
            
            # Simulate random door status changes (rare)
            if np.random.random() < 0.01:  # 1% chance each interval
                is_open = np.random.random() < 0.3  # 30% chance of being open
                socketio.emit('door_status', {'isOpen': is_open})
                if is_open:
                    socketio.emit('new_alert', {
                        'timestamp': time.strftime('%H:%M:%S'),
                        'message': 'Fridge door opened!'
                    })
            
            # Simulate power status changes (very rare)
            if np.random.random() < 0.001:  # 0.1% chance each interval
                is_on = np.random.random() < 0.9  # 90% chance of being on
                socketio.emit('power_status', {'isOn': is_on})
                if not is_on:
                    socketio.emit('new_alert', {
                        'timestamp': time.strftime('%H:%M:%S'),
                        'message': 'WARNING: Fridge power disconnected!'
                    })
        
        time.sleep(1)  # Update every second

# Start the fridge simulation thread when the app starts
fridge_simulation_thread = threading.Thread(target=simulate_fridge_updates, daemon=True)
fridge_simulation_thread.start()

def update_inventory(detections):
    """Update the inventory based on new detections"""
    global current_inventory
    current_inventory = detections

@socketio.on('update_inventory')
def handle_update_inventory(data):
    """Handle manual inventory updates from the client"""
    global current_inventory
    if 'inventory' in data:
        current_inventory = data['inventory']
        # Broadcast the update to all clients
        socketio.emit('inventory_updated', {'inventory': current_inventory}, room='viewers')
    return {'status': 'success'}

@socketio.on('get_inventory')
def handle_get_inventory(data=None):
    """Handle request for current inventory state"""
    return {'inventory': current_inventory}

@app.route('/recipes')
@login_required
def recipes():
    return render_template('recipes.html')

def update_sensor_data():
    """Background thread to update sensor data"""
    global sensor_data
    
    # Define thresholds
    TEMP_THRESHOLD = 8  # °C
    HUMIDITY_THRESHOLD = 60  # %
    MQ135_BASELINE = 110  # ppm
    MQ2_BASELINE = 298    # ppm
    GAS_WARNING_THRESHOLD = 30  # ppm above baseline
    GAS_DANGER_THRESHOLD =  60 # ppm above baseline
    
    while True:
        try:
            # Read sensor data, default to 0 if sensor read fails
            try:
                temp, humidity, mq135, mq2 = sensor_monitor.read_sensor_data()
            except:
                temp, humidity, mq135, mq2 = 0, 0, 0, 0
                logger.warning("Failed to read sensor data, using default values (0)")
            
            # Calculate warning levels
            temp_status = 'normal' if temp <= TEMP_THRESHOLD else 'warning'
            humidity_status = 'normal' if humidity <= HUMIDITY_THRESHOLD else 'warning'
            
            # Calculate gas warning levels for MQ135
            mq135_status = 'normal'
            mq135_warning_level = MQ135_BASELINE + GAS_WARNING_THRESHOLD  # depends on surrounding
            mq135_danger_level = MQ135_BASELINE + GAS_DANGER_THRESHOLD   # depends on surrounding
            
            if mq135 > 0:  # Only if sensor is working
                if mq135 >= mq135_danger_level:
                    mq135_status = 'danger'
                elif mq135 >= mq135_warning_level:
                    mq135_status = 'warning'
                elif mq135 > 130:  # Fridge condition threshold
                    mq135_status = 'high'
            
            # Calculate gas warning levels for MQ2
            mq2_status = 'normal'
            mq2_warning_level = MQ2_BASELINE + GAS_WARNING_THRESHOLD # depends on surrounding
            mq2_danger_level = MQ2_BASELINE + GAS_DANGER_THRESHOLD   # depends on surrounding
            
            if mq2 > 0:  # Only if sensor is working
                if mq2 >= mq2_danger_level:
                    mq2_status = 'danger'
                elif mq2 >= mq2_warning_level:
                    mq2_status = 'warning'
                elif mq2 > 280:  # Fridge condition threshold
                    mq2_status = 'high'
            
            # Update door status (simulated based on humidity)
            door_status = 'closed' if humidity < 50 else 'open'
            
            # Update sensor data with status levels
            sensor_data.update({
                'temperature': temp,
                'temp_status': temp_status,
                'humidity': humidity,
                'humidity_status': humidity_status,
                'mq135': mq135,
                'mq135_status': mq135_status,
                'mq2': mq2,
                'mq2_status': mq2_status,
                'door_status': door_status,
                'power_status': 'on',
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Emit update to all connected clients
            socketio.emit('sensor_update', sensor_data)
            
            # Generate alerts for warning conditions
            if temp > 0 and temp_status == 'warning':  # Only alert if sensor is working
                socketio.emit('alert', {
                    'type': 'warning',
                    'message': f'High temperature detected: {temp}°C'
                })
            
            if humidity > 0 and humidity_status == 'warning':  # Only alert if sensor is working
                socketio.emit('alert', {
                    'type': 'warning',
                    'message': f'High humidity detected: {humidity}%'
                })
            
            if mq135 > 0:  # Only alert if sensor is working
                if mq135_status == 'danger':
                    socketio.emit('alert', {
                        'type': 'danger',
                        'message': f'Dangerous NH3/CO2 levels detected: {mq135} ppm '
                    })
                elif mq135_status == 'warning':
                    socketio.emit('alert', {
                        'type': 'warning',
                        'message': f'Elevated NH3/CO2 levels detected: {mq135} ppm '
                    })
                elif mq135_status == 'high':
                    socketio.emit('alert', {
                        'type': 'high',
                        'message': f'High NH3/CO2 levels detected in fridge: {mq135} ppm '
                    })
            
            if mq2 > 0:  # Only alert if sensor is working
                if mq2_status == 'danger':
                    socketio.emit('alert', {
                        'type': 'danger',
                        'message': f'Dangerous flammable gas levels detected: {mq2} ppm '
                    })
                elif mq2_status == 'warning':
                    socketio.emit('alert', {
                        'type': 'warning',
                        'message': f'Elevated flammable gas levels detected: {mq2} ppm '
                    })
                elif mq2_status == 'high':
                    socketio.emit('alert', {
                        'type': 'high',
                        'message': f'High flammable gas levels detected in fridge: {mq2} ppm '
                    })
            
            # Add delay
            socketio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in sensor update thread: {str(e)}")
            socketio.sleep(1)

# Start sensor monitoring thread when the app starts
sensor_thread = None

# Replace @app.before_first_request with initialization flag
_sensor_thread_started = False

@app.before_request
def start_sensor_thread():
    """Start the sensor monitoring thread before first request"""
    global sensor_thread, _sensor_thread_started
    if not _sensor_thread_started:
        if not sensor_thread or not sensor_thread.is_alive():
            sensor_thread = socketio.start_background_task(update_sensor_data)
        _sensor_thread_started = True

@socketio.on('request_sensor_data')
def handle_sensor_data_request():
    """Handle client request for current sensor data"""
    emit('sensor_update', sensor_data)

@app.route('/get_sensor_data')
def get_sensor_data():
    """API endpoint to get current sensor data"""
    return jsonify(sensor_data)

@socketio.on('request_inventory_update')
def handle_inventory_request():
    """Handle request for inventory data with current conditions"""
    # Get current sensor readings
    current_conditions = sensor_data if 'sensor_data' in globals() else {
        'temperature': 0,
        'humidity': 0,
        'mq135': 0,
        'mq2': 0
    }
    
    # Emit both inventory and sensor data
    socketio.emit('inventory_update', {
        'inventory': current_inventory,
        'conditions': current_conditions
    })

def get_sensor_status(value, sensor_type, condition):
    """Get status for a sensor value based on type and condition"""
    if condition == 'room':
        if sensor_type == 'temperature':
            if value < 20:
                return 'low'
            elif value > 30:
                return 'high'
            else:
                return 'normal'
        elif sensor_type == 'humidity':
            if value < 30:
                return 'low'
            elif value > 60:
                return 'high'
            else:
                return 'normal'
        elif sensor_type == 'mq135':
            if value < 100:
                return 'low'
            elif value > 200:
                return 'high'
            else:
                return 'normal'
        elif sensor_type == 'mq2':
            if value < 200:
                return 'low'
            elif value > 300:
                return 'high'
            else:
                return 'normal'
    else:  # fridge condition
        if sensor_type == 'temperature':
            if value < 2:
                return 'low'
            elif value > 8:
                return 'high'
            else:
                return 'normal'
        elif sensor_type == 'humidity':
            if value < 40:
                return 'low'
            elif value > 80:
                return 'high'
            else:
                return 'normal'
        elif sensor_type == 'mq135':
            if value < 100:
                return 'low'
            elif value > 130:
                return 'high'  # This will show red bar
            else:
                return 'normal'
        elif sensor_type == 'mq2':
            if value < 200:
                return 'low'
            elif value > 280:
                return 'high'  # This will show red bar
            else:
                return 'normal'

if __name__ == '__main__':
    try:
        # Create output directory at startup
        output_dir = Path('detected_fruits')
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory created/verified at: {output_dir.absolute()}")
        
        # Run without watchdog reloader
        socketio.run(
            app,
            debug=False,
            host='127.0.0.1',
            port=5000,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise 

    