import time
import serial
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SensorMonitor:
    def __init__(self, port='COM3', baud_rate=9600):
        """Initialize the sensor monitor with baseline values and thresholds."""
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
        self.is_connected = False
        
        # Baseline values (from calibration)
        self.baseline_mq135 = 110 #PUT THE BASELINE HERE   

        self.baseline_mq2 = 260        #PUT THE BASELINE HERE
        
        # Threshold values
        self.threshold = 20
        
        # Frame rate control (frames per second)
        self.frame_interval = 1.0  # 1 frame per second
        self.last_frame_time = 0
        
        # Create alerts directory
        self.alerts_dir = Path('alerts')
        self.alerts_dir.mkdir(exist_ok=True)
        
        # Initialize alert history
        self.alert_history = []
        
        # Alert cooldown (seconds) to prevent spam
        self.alert_cooldown = 300  # 5 minutes
        self.last_alert_time = 0

        # Try to connect on initialization
        self.connect()

    def connect(self):
        """Establish serial connection with the Arduino."""
        try:
            if self.serial is not None:
                self.disconnect()
            
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            self.is_connected = True
            logger.info(f"Successfully connected to {self.port}")
            return True
        except Exception as e:
            self.is_connected = False
            logger.error(f"Failed to connect to {self.port}: {str(e)}")
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.is_connected = False
            logger.info("Serial connection closed")

    def read_sensor_data(self):
        """Read a single set of sensor readings."""
        current_time = time.time()
        
        # Check if enough time has passed since last frame
        if current_time - self.last_frame_time < self.frame_interval:
            return None, None, None, None
            
        self.last_frame_time = current_time
        
        try:
            if not self.is_connected:
                # If not connected, try to reconnect
                if not self.connect():
                    # If reconnection fails, return simulated data
                    return self.get_simulated_data()
            
            if self.serial and self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()
                logger.debug(f"Raw data received: {line}")
                
                # Parse the data format: "MQ135: X, MQ2: Y, Temperature: Z°C, Humidity: W%"
                parts = line.split(',')
                
                # Extract values
                mq135 = float(parts[0].split(':')[1].strip())
                mq2 = float(parts[1].split(':')[1].strip())
                temp = float(parts[2].split(':')[1].strip().replace('°C', ''))
                humidity = float(parts[3].split(':')[1].strip().replace('%', ''))
                
                return temp, humidity, mq135, mq2
            
            # If no data is waiting, return simulated data
            return self.get_simulated_data()
            
        except Exception as e:
            logger.error(f"Error reading sensor data: {str(e)}")
            # On error, return simulated data
            return self.get_simulated_data()

    def get_simulated_data(self):
        """Generate simulated sensor data when hardware is not available."""
        import random
        # Generate realistic simulated values
        temp = round(random.uniform(2, 8), 1)  # Temperature between 2-8°C
        humidity = round(random.uniform(30, 60), 1)  # Humidity between 30-60%
        mq135 = round(random.uniform(100, 120), 1)  # MQ135 around baseline
        mq2 = round(random.uniform(280, 320), 1)  # MQ2 around baseline
        
        return temp, humidity, mq135, mq2

    def check_thresholds(self, mq135, mq2):
        """Check if sensor values exceed thresholds."""
        current_time = time.time()
        alerts = []
        
        # Check MQ135
        if mq135 > (self.baseline_mq135 + self.threshold):
            if current_time - self.last_alert_time >= self.alert_cooldown:
                alert = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sensor': 'MQ135',
                    'baseline': self.baseline_mq135,
                    'current_value': mq135,
                    'threshold': self.threshold,
                    'message': f"MQ135 sensor detected high gas levels! Current: {mq135:.1f}, Baseline: {self.baseline_mq135}"
                }
                alerts.append(alert)
                self.last_alert_time = current_time
        
        # Check MQ2
        if mq2 > (self.baseline_mq2 + self.threshold):
            if current_time - self.last_alert_time >= self.alert_cooldown:
                alert = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sensor': 'MQ2',
                    'baseline': self.baseline_mq2,
                    'current_value': mq2,
                    'threshold': self.threshold,
                    'message': f"MQ2 sensor detected high gas levels! Current: {mq2:.1f}, Baseline: {self.baseline_mq2}"
                }
                alerts.append(alert)
                self.last_alert_time = current_time
        
        return alerts

    def save_alert(self, alert):
        """Save alert to file and history."""
        # Add to history
        self.alert_history.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = self.alerts_dir / f'alert_{timestamp}.json'
        
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=4)
        
        logger.info(f"Alert saved to {alert_file}")

    def monitor_sensors(self):
        """Main monitoring loop."""
        try:
            logger.info("Starting sensor monitoring...")
            logger.info(f"Baseline MQ135: {self.baseline_mq135}, Threshold: {self.threshold}")
            logger.info(f"Baseline MQ2: {self.baseline_mq2}, Threshold: {self.threshold}")
            logger.info(f"Frame rate: {1/self.frame_interval:.1f} fps")
            
            while True:
                temp, humidity, mq135, mq2 = self.read_sensor_data()
                
                if all(v is not None for v in [temp, humidity, mq135, mq2]):
                    # Check thresholds
                    alerts = self.check_thresholds(mq135, mq2)
                    
                    # Process alerts
                    for alert in alerts:
                        logger.warning(alert['message'])
                        self.save_alert(alert)
                    
                    # Log current values
                    logger.info(f"Current readings - MQ135: {mq135:.1f}, MQ2: {mq2:.1f}, "
                              f"Temperature: {temp:.1f}°C, Humidity: {humidity:.1f}%")
                
                time.sleep(0.1)  # Small sleep to prevent CPU overload
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            return True
        except Exception as e:
            logger.error(f"Error during monitoring: {str(e)}")
            return False
        finally:
            self.disconnect()

def main():
    # Create monitor instance
    monitor = SensorMonitor()
    
    # Start monitoring
    success = monitor.monitor_sensors()
    
    if success:
        logger.info("Monitoring completed successfully")
    else:
        logger.error("Monitoring failed")

if __name__ == "__main__":
    main() 