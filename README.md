# Crack Detection Project


Welcome to the Crack Detection Project repository! This project demonstrates a complete pipeline for crack detection in concrete structures using deep learning. The repository is organized into modular components showcasing data preparation, model building with both TensorFlow and PyTorch, and evaluation. It is an excellent reference for understanding how to integrate classical computer vision techniques with modern deep learning frameworks such as GANs, ResNet50, and ResNet18.

Table of Contents
Project Overview
Directory Structure
Installation
Usage
Data Preparation
Training with TensorFlow (ResNet50)
Training with PyTorch (Pre-trained ResNet18)
Project Modules
Contributing
License
Contact
Project Overview
Crack detection is crucial for structural health monitoring and the maintenance of concrete structures. In this project, we develop a deep learning pipeline to detect cracks by classifying images as either containing cracks (positive) or not (negative). The solution leverages two primary approaches:

TensorFlow/Keras Implementation:
Building a classifier using ResNet50 with an ImageDataGenerator for data augmentation and training.

PyTorch Implementation:
Using pre-trained ResNet18 for transfer learning. This approach includes modifying the model’s final layer, training the network, and identifying misclassified samples.

In addition, the repository includes modules for data preparation and creating a custom PyTorch Dataset for efficient loading of tensorized image data.

Directory Structure
The repository is organized into the following files:

```plaintext
crack_detection/
├── README.md                     # This file
├── requirements.txt              # List of required packages
├── data_preparation.py           # Script for image data listing and visualization
├── data_preparation_pytorch.py   # Script for PyTorch dataset object creation and sample visualization
├── classifier_resnet50.py        # TensorFlow/Keras ResNet50 model building and training script
├── pretrained_models.py          # PyTorch pre-trained ResNet18 model, training, and evaluation script
├── dataset.py                    # Custom PyTorch Dataset class for crack detection
└── utils.py                      # Utility functions (e.g., plotting helpers)
```

Installation
1. Clone the Repository:
```bash
git clone https://github.com/yourusername/crack_detection.git
cd crack_detection
```
2. Create and Activate a Virtual Environment (Optional but Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install the Required Packages:
```bash
pip install -r requirements.txt
```
The main dependencies include:

PyTorch and Torchvision
TensorFlow and Keras
Matplotlib
Pillow

Usage
Data Preparation
The data_preparation.py script demonstrates how to download, preprocess, and visualize the raw image data. It includes examples for listing image file paths and plotting samples of both cracked and non-cracked concrete.
```bash
python data_preparation.py
```
Training with TensorFlow (ResNet50)
The classifier_resnet50.py module builds a ResNet50-based classifier using TensorFlow/Keras. It uses an ImageDataGenerator for real-time data augmentation and trains the model for a specified number of epochs.
```bash
python classifier_resnet50.py
```
The trained model is saved as classifier_resnet_model.h5.

Training with PyTorch (Pre-trained ResNet18)
The pretrained_models.py script demonstrates how to:

Load a pre-trained ResNet18 model.
Freeze its parameters and modify the final layer for binary classification.
Train the model on a custom PyTorch Dataset.
Identify and display misclassified samples from the validation data.
```bash
python pretrained_models.py
```
Data Loading with PyTorch
The data_preparation_pytorch.py module shows how to create and use the custom PyTorch dataset. It displays specific samples from the validation dataset, which helps in visual inspection of data and model predictions.
```bash
python data_preparation_pytorch.py
```
Project Modules
dataset.py: Contains the custom CrackDataset class for loading and processing tensor data.
data_preparation.py: Handles raw image data preparation and visualization.
data_preparation_pytorch.py: Demonstrates the use of the PyTorch dataset and displays validation samples.
classifier_resnet50.py: Implements a ResNet50-based classifier using TensorFlow/Keras.
pretrained_models.py: Implements a pre-trained ResNet18 model using PyTorch, including training and misclassification analysis.
utils.py: Provides helper functions (e.g., for plotting training loss curves).









