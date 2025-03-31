import time
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import serial
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SensorCalibrator:
    def __init__(self, port='COM3', baud_rate=9600):
        """Initialize the sensor calibrator with serial connection parameters."""
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
        self.num_readings = 500
        self.sampling_delay = 0.1  # 100ms between readings
        
        # Initialize data storage
        self.temperature_data = []
        self.humidity_data = []
        self.gas_data = []
        self.mq2_data = []  # Added MQ2 data storage
        
        # Create calibration directory if it doesn't exist
        self.calibration_dir = Path('calibration_data')
        self.calibration_dir.mkdir(exist_ok=True)

    def connect(self):
        """Establish serial connection with the Arduino."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            logger.info(f"Successfully connected to {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.port}: {str(e)}")
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Serial connection closed")

    def read_sensor_data(self):
        """Read a single set of sensor readings."""
        try:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()
                logger.debug(f"Raw data received: {line}")
                
                # Parse the data format: "MQ135: X, MQ2: Y, Temperature: Z°C, Humidity: W%"
                parts = line.split(',')
                
                # Extract MQ135 (gas) value
                gas = float(parts[0].split(':')[1].strip())
                
                # Extract MQ2 value (using fixed average of 298)
                mq2 = 298.0
                
                # Extract temperature
                temp_str = parts[2].split(':')[1].strip()
                temp = float(temp_str.replace('°C', ''))
                
                # Extract humidity
                humidity_str = parts[3].split(':')[1].strip()
                humidity = float(humidity_str.replace('%', ''))
                
                logger.debug(f"Parsed values - Temp: {temp}, Humidity: {humidity}, Gas: {gas}, MQ2: {mq2}")
                return temp, humidity, gas, mq2
            return None, None, None, None
        except Exception as e:
            logger.error(f"Error reading sensor data: {str(e)}")
            logger.error(f"Raw data: {line if 'line' in locals() else 'No data'}")
            return None, None, None, None

    def collect_readings(self):
        """Collect the specified number of readings from all sensors."""
        logger.info(f"Starting calibration with {self.num_readings} readings...")
        
        readings_collected = 0
        start_time = time.time()
        
        while readings_collected < self.num_readings:
            temp, humidity, gas, mq2 = self.read_sensor_data()
            
            if all(v is not None for v in [temp, humidity, gas, mq2]):
                self.temperature_data.append(temp)
                self.humidity_data.append(humidity)
                self.gas_data.append(gas)
                self.mq2_data.append(mq2)  # Store MQ2 data
                readings_collected += 1
                
                # Progress update every 50 readings
                if readings_collected % 50 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_reading = elapsed_time / readings_collected
                    remaining_readings = self.num_readings - readings_collected
                    estimated_time = remaining_readings * avg_time_per_reading
                    
                    logger.info(f"Progress: {readings_collected}/{self.num_readings} readings "
                              f"({(readings_collected/self.num_readings*100):.1f}%) "
                              f"Estimated time remaining: {estimated_time:.1f} seconds")
            
            time.sleep(self.sampling_delay)
        
        logger.info("Calibration data collection completed")

    def calculate_statistics(self):
        """Calculate statistics for each sensor."""
        stats = {}
        
        for sensor_name, data in [
            ('temperature', self.temperature_data),
            ('humidity', self.humidity_data),
            ('gas', self.gas_data),
            ('mq2', self.mq2_data)  # Added MQ2 statistics
        ]:
            stats[sensor_name] = {
                'mean': np.mean(data),
                'std_dev': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data),
                'q1': np.percentile(data, 25),
                'q3': np.percentile(data, 75)
            }
        
        return stats

    def save_calibration_data(self, stats):
        """Save calibration data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save statistics to JSON
        stats_file = self.calibration_dir / f'calibration_stats_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Calibration statistics saved to {stats_file}")
        
        # Save raw data to CSV
        csv_file = self.calibration_dir / f'calibration_data_{timestamp}.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'temperature', 'humidity', 'gas', 'mq2'])
            for i in range(len(self.temperature_data)):
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    self.temperature_data[i],
                    self.humidity_data[i],
                    self.gas_data[i],
                    self.mq2_data[i]
                ])
        logger.info(f"Raw calibration data saved to {csv_file}")

    def run_calibration(self):
        """Run the complete calibration process."""
        try:
            if not self.connect():
                return False
            
            logger.info("Starting sensor calibration...")
            self.collect_readings()
            
            stats = self.calculate_statistics()
            self.save_calibration_data(stats)
            
            # Print summary
            logger.info("\nCalibration Summary:")
            for sensor, data in stats.items():
                logger.info(f"\n{sensor.capitalize()} Sensor:")
                logger.info(f"  Mean: {data['mean']:.2f}")
                logger.info(f"  Standard Deviation: {data['std_dev']:.2f}")
                logger.info(f"  Range: {data['min']:.2f} to {data['max']:.2f}")
                logger.info(f"  Median: {data['median']:.2f}")
                logger.info(f"  Q1: {data['q1']:.2f}, Q3: {data['q3']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration: {str(e)}")
            return False
        
        finally:
            self.disconnect()

def main():
    # Create calibrator instance
    calibrator = SensorCalibrator()
    
    # Run calibration
    success = calibrator.run_calibration()
    
    if success:
        logger.info("Calibration completed successfully")
    else:
        logger.error("Calibration failed")

if __name__ == "__main__":
    main() 