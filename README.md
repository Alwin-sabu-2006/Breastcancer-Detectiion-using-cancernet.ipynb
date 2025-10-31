# CancerNet: Histopathology Image Classifier
This project was created as part of my **Internship** with Internforte.
This project is a Jupyter/Google Colab notebook that builds, trains, and evaluates a Convolutional Neural Network (CNN) to classify breast cancer histopathology images as **Benign (Class 0)** or **Malignant (Class 1)**.

The notebook demonstrates a complete machine learning workflow: data loading, preprocessing, model definition, training, and evaluation.

## 🚀 Project Overview

The goal of this project is to create a binary image classifier for medical diagnostics. It uses a custom CNN model, dubbed "CancerNet," built with TensorFlow and Keras. The model learns to distinguish between benign and malignant tissue samples from small image patches (50x50 pixels) derived from larger histopathology slides.

This notebook is an excellent example of applying deep learning to a real-world computer vision problem in the medical field.

### Features
* **Data Loading:** Loads images from a directory and intelligently extracts class labels (`0` or `1`) from the filenames.
* **Preprocessing:** Resizes images to a uniform (50, 50, 3) shape and normalizes pixel values to be between 0 and 1.
* **CNN Model:** Defines a "CancerNet" architecture from scratch using `Conv2D`, `MaxPooling2D`, and `Dense` layers.
* **Model Training:** Trains the model on the image dataset, splitting it into training and validation sets.
* **Performance Evaluation:** After training, the model's performance is measured on a held-out test set, generating:
    * An **Accuracy vs. Epoch** plot
    * A **Classification Report** (with precision, recall, f1-score)
    * A visual **Confusion Matrix**

## 🛠️ Technology Stack

* **Python 3**
* **Google Colab / Jupyter Notebook**
* **TensorFlow (Keras):** For building and training the deep learning model.
* **Scikit-learn:** For splitting data (`train_test_split`) and generating evaluation metrics (`classification_report`, `confusion_matrix`).
* **OpenCV (`opencv-python`):** For loading and resizing images.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For plotting the results.

## 🧠 Model Architecture

The "CancerNet" model is a sequential CNN with the following structure:

1.  **Block 1:** `Conv2D` (32 filters, 3x3 kernel, ReLU) -> `MaxPooling2D`
    * *Purpose:* Finds simple features like edges.
2.  **Block 2:** `Conv2D` (64 filters, 3x3 kernel, ReLU) -> `MaxPooling2D`
    * *Purpose:* Combines simple features into more complex shapes.
3.  **Block 3:** `Conv2D` (128 filters, 3x3 kernel, ReLU) -> `MaxPooling2D`
    * *Purpose:* Detects more abstract patterns and textures.
4.  **Classifier Head:**
    * `Flatten()`: Converts the 3D feature maps into a 1D vector.
    * `Dense(128, 'relu')`: A fully-connected layer for final classification logic.
    * `Dropout(0.5)`: A regularization technique to prevent overfitting.
    * `Dense(1, 'sigmoid')`: The output layer. It produces a single probability score (0 for Benign, 1 for Malignant).

## ⚙️ How to Use

### 1. Prerequisites
You can run this project directly in Google Colab. If running locally, ensure you have the required libraries installed:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn
