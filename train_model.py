import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
import shutil

# Define constants
IMAGE_SIZE = (224, 224) # MobileNetV2 expects 224x224
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'dataset/Training'
TEST_DIR = 'dataset/Testing'
MODEL_PATH = 'brain_tumor_model.h5'

def create_synthetic_data():
    print("Generating synthetic data for demonstration...")
    base_dir = 'synthetic_data'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    for category in ['train', 'validation']:
        for label in ['yes', 'no']:
            path = os.path.join(base_dir, category, label)
            os.makedirs(path, exist_ok=True)
            # Generate 10 random images per class
            for i in range(10):
                img_array = np.random.rand(224, 224, 3) * 255
                img = Image.fromarray(img_array.astype('uint8'))
                img.save(os.path.join(path, f'img_{i}.jpg'))
    return base_dir

def build_model():
    # Use MobileNetV2 as base for better feature extraction potential
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, notumor, pituitary
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Changed from binary_crossentropy
                  metrics=['accuracy'])
    return model

def train_model():
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory '{TRAIN_DIR}' not found.")
        print("Please ensure the dataset is available at 'dataset/Training'")
        return

    print(f"Using dataset from '{TRAIN_DIR}'")
    print(f"Training 4-class brain tumor classifier: glioma, meningioma, notumor, pituitary")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        validation_split=0.2  # 20% for validation
    )

    try:
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',  # Changed from binary
            subset='training'
        )

        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',  # Changed from binary
            subset='validation'
        )

        print(f"\nFound {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Classes: {train_generator.class_indices}")

        model = build_model()
        model.summary()
        
        print("\nStarting training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            verbose=1
        )

        model.save(MODEL_PATH)
        print(f"\n✅ Model saved successfully as {MODEL_PATH}")
        
        # Print final accuracy
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nFinal Training Accuracy: {final_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()
