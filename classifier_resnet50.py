# classifier_resnet50.py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_classifier(input_shape=(224, 224, 3), num_classes=2):
    # Load base model ResNet50 with imagenet weights, excluding top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add final dense layer for classification
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    # Directories for training and validation (update these paths as needed)
    train_dir = "./data/train"
    validation_dir = "./data/validation"
    
    # Create ImageDataGenerators with basic rescaling
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 32  # adjust batch size if needed
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Build and compile the model
    model = build_classifier(input_shape=(224,224,3), num_classes=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # print the model architecture

    steps_per_epoch_training = len(train_generator)
    steps_per_epoch_validation = len(validation_generator)
    num_epochs = 2

    # Train the model using fit_generator (for TF<2.0) or fit in newer TF versions
    fit_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch_training,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_validation,
        verbose=1,
    )

    # Save the trained model
    model.save('classifier_resnet_model.h5')
    print("Model saved as classifier_resnet_model.h5")

if __name__ == "__main__":
    main()
