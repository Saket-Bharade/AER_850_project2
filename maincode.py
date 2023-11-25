from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import pandas as pd


import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

# Path to the ZIP file containing images
zip_file_path = 'path/to/your/images.zip'

# Directory to extract the images
extracted_dir = 'extracted_images'

# Extract images from the ZIP file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Function to organize images into train, validation, and test directories
def organize_images(data_dir, extracted_dir, test_size=0.2, random_state=42):
    all_images = os.listdir(extracted_dir)
    train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=random_state)
    val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=random_state)

    for subset, subset_images in zip(['Train', 'Validation', 'Test'], [train_images, val_images, test_images]):
        subset_dir = os.path.join(data_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for image in subset_images:
            src_path = os.path.join(extracted_dir, image)
            dest_path = os.path.join(subset_dir, image)
            os.rename(src_path, dest_path)

# Organize images into train, validation, and test directories
organize_images(data_dir='.', extracted_dir=extracted_dir)

# Continue with the rest of your code...
# (ImageDataGenerator, model definition, compilation, training, etc.)

# Update data directories
train_data_dir = 'Train'
val_data_dir = 'Validation'
test_data_dir = 'Test'




# Define input image shape
input_shape = (100, 100, 3)

# Establish data directories
train_data_dir = 'Train'
val_data_dir = 'Validation'
test_data_dir = 'Test'

# Define data generators with augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data generator with rescaling only
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Build a simple model (you can replace this with your own model architecture)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Assuming 3 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# If you have a test set, you can evaluate the model using model.evaluate
# test_generator = val_datagen.flow_from_directory(test_data_dir, ...)
# test_loss, test_acc = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_acc}")
