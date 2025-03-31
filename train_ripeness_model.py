import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Check for GPU availability
print("\nTensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("GPU Device Name:", tf.test.gpu_device_name())
    # Enable memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(e)
else:
    print("No GPU found. Training will be slower on CPU.")

class RipenessModelTrainer:
    def __init__(self, data_dir='dataset', img_size=224, batch_size=64):  # Increased batch size for GPU
        self.data_dir = Path(data_dir).resolve()  # Get absolute path
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Define fruits and conditions
        self.fruits = ['Apple', 'Banana', 'Orange', 'Mango']
        self.conditions = ['Fresh', 'Rotten', 'Formalin-mixed']
        
        print(f"\nDataset directory: {self.data_dir}")
        print(f"Directory exists: {self.data_dir.exists()}")
        
        # Enhanced data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        self.val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Enable mixed precision training for better GPU performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def create_data_generators(self):
        print("\nChecking dataset structure:")
        total_images = 0
        class_counts = {}
        
        for fruit in self.fruits:
            fruit_dir = self.data_dir / fruit
            print(f"\nChecking {fruit} directory:")
            print(f"Directory exists: {fruit_dir.exists()}")
            if fruit_dir.exists():
                class_total = 0
                for condition in self.conditions:
                    condition_dir = fruit_dir / condition
                    print(f"  {condition}: {condition_dir.exists()}")
                    if condition_dir.exists():
                        num_images = len(list(condition_dir.glob('*.jpg')))
                        print(f"    Number of images: {num_images}")
                        class_total += num_images
                class_counts[fruit] = class_total
                total_images += class_total
        
        print(f"\nTotal images found: {total_images}")
        print("\nClass counts:")
        for fruit, count in class_counts.items():
            print(f"{fruit}: {count}")
        
        if total_images == 0:
            raise ValueError("No images found in the dataset directory!")
        
        print("\nCreating data generators...")
        # Create training generator
        self.train_generator = self.train_datagen.flow_from_directory(
            str(self.data_dir),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.fruits
        )
        
        # Create validation generator
        self.val_generator = self.val_datagen.flow_from_directory(
            str(self.data_dir),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.fruits
        )
        
        # Get the class indices from the generator
        self.class_indices = self.train_generator.class_indices
        
        print("\nClass mapping:")
        for class_name, class_index in self.class_indices.items():
            print(f"{class_name}: {class_index}")
        
        # Calculate class weights after generators are created
        self.class_weights = {}
        for fruit, count in class_counts.items():
            self.class_weights[self.class_indices[fruit]] = total_images / (len(self.fruits) * count)
        
        print("\nClass weights:")
        for class_idx, weight in self.class_weights.items():
            print(f"Class {class_idx}: {weight}")
    
    def create_model(self):
        print("\nCreating model...")
        # Load the pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom layers with increased capacity
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.fruits), activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        print("Model created successfully")
    
    def train_model(self, epochs=30):
        print("\nCompiling model...")
        # Compile the model with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=True  # Enable XLA compilation for better GPU performance
        )
        
        print("\nStarting training...")
        # Train the model with class weights and GPU optimizations
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            class_weight=self.class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=7,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ],
            workers=4,  # Increased number of workers for better GPU utilization
            use_multiprocessing=True  # Enable multiprocessing for data loading
        )
    
    def plot_training_history(self):
        print("\nPlotting training history...")
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved")
    
    def save_model(self, model_path):
        print("\nSaving model...")
        # Save the trained model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save class indices
        import json
        with open('class_indices.json', 'w') as f:
            json.dump(self.class_indices, f)
        print("Class indices saved to class_indices.json")

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Initialize trainer
    trainer = RipenessModelTrainer(
        data_dir='dataset',
        img_size=224,
        batch_size=64
    )
    
    # Create data generators
    trainer.create_data_generators()
    
    # Create and train model
    trainer.create_model()
    trainer.train_model(epochs=30)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save the model
    trainer.save_model('ripeness_model.h5')

if __name__ == "__main__":
    main() 