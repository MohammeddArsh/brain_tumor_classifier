# Brain Tumor MRI Classification using CNN

CNN-based deep learning system for classifying brain MRI images into four tumor categories with high accuracy.

## Overview
This project implements convolutional neural network (CNN) models to automatically classify brain MRI scans
into four distinct tumor classes. The objective is to support medical image analysis through accurate
and efficient classification.

## Dataset
- Source: Kaggle Brain Tumor MRI Dataset
- Data type: MRI images
- Number of classes: 4

## Methodology
- Image preprocessing and normalization
- Design and training of CNN architectures
- Performance evaluation and comparison across models
- Selection of best-performing model based on accuracy

## Results
- Best classification accuracy achieved: **99.39%**

## Deployment
The trained CNN model was deployed locally using **FastAPI**, enabling image upload and real-time
prediction through a simple web interface.

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, pandas
- FastAPI

## How to Run
1. Install required dependencies.
2. Run the FastAPI server.
3. Upload an MRI image through the API endpoint to obtain predictions.

## Publication
A research paper based on this project was published at the  
**13th International Conference on Interdisciplinary Research for Sustainable Development (ICIRSD), Jamia Hamdard**.
