import os
import shutil
from pathlib import Path

def create_dataset_structure():
    # Create main dataset directory
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    # Define fruits and their conditions
    fruits = ['Apple', 'Banana', 'Orange']
    conditions = ['Fresh', 'Rotten', 'Formalin-mixed']
    
    # Create nested directory structure
    for fruit in fruits:
        fruit_dir = dataset_dir / fruit
        fruit_dir.mkdir(exist_ok=True)
        
        for condition in conditions:
            condition_dir = fruit_dir / condition
            condition_dir.mkdir(exist_ok=True)
    
    print("Dataset structure created successfully!")
    print("\nPlease organize your images as follows:")
    print("dataset/")
    print("├── Apple/")
    print("│   ├── Fresh/")
    print("│   │   ├── image1.jpg")
    print("│   │   └── ...")
    print("│   ├── Rotten/")
    print("│   │   ├── image1.jpg")
    print("│   │   └── ...")
    print("│   └── Formalin-mixed/")
    print("│       ├── image1.jpg")
    print("│       └── ...")
    print("├── Banana/")
    print("│   ├── Fresh/")
    print("│   │   ├── image1.jpg")
    print("│   │   └── ...")
    print("│   ├── Rotten/")
    print("│   │   ├── image1.jpg")
    print("│   │   └── ...")
    print("│   └── Formalin-mixed/")
    print("│       ├── image1.jpg")
    print("│       └── ...")
    print("└── Orange/")
    print("    ├── Fresh/")
    print("    │   ├── image1.jpg")
    print("    │   └── ...")
    print("    ├── Rotten/")
    print("    │   ├── image1.jpg")
    print("    │   └── ...")
    print("    └── Formalin-mixed/")
    print("        ├── image1.jpg")
    print("        └── ...")
    
    print("\nInstructions for collecting images:")
    print("1. For each fruit (Apple, Banana, Orange):")
    print("   - Take multiple photos of fresh samples")
    print("   - Take multiple photos of rotten samples")
    print("   - Take multiple photos of formalin-mixed samples")
    print("2. Try to capture images from different angles")
    print("3. Ensure good lighting conditions")
    print("4. Include both close-up and full fruit views")
    print("5. Aim for at least 50 images per condition per fruit")
    print("6. Save images in JPG or PNG format")
    print("7. Make sure images are clear and well-focused")

if __name__ == "__main__":
    create_dataset_structure() 