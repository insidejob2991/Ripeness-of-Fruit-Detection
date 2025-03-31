import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def main():
    # Create a directory for YOLO files if it doesn't exist
    if not os.path.exists('yolo_files'):
        os.makedirs('yolo_files')
    
    # URLs for YOLO files
    urls = {
        'yolo_files/yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'yolo_files/yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'yolo_files/coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    # Download each file
    for filename, url in urls.items():
        download_file(url, filename)
    
    print("\nAll files downloaded successfully!")

if __name__ == "__main__":
    main() 