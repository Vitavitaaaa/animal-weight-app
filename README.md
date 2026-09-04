Animal Weight Estimation System
Overview

This project is an information system for automatic cattle weight estimation using computer vision and deep learning technologies.

The system analyzes RGB images of cattle and predicts body weight without requiring physical contact or specialized hardware.

Features
Image classification using ResNet18
Animal segmentation using U-Net
Weight prediction using a dual-stream neural network
Web-based user interface
REST API built with Flask
Real-time inference from uploaded images
Architecture

The system consists of:

Frontend (HTML, CSS, JavaScript)
Backend API (Flask)
Classification Module (ResNet18)
Segmentation Module (U-Net)
Weight Prediction Module (ResNet34 + U-Net Encoder)
Model Storage

Processing pipeline:

Image Upload → Classification → Segmentation Feature Extraction → Weight Prediction → Result Display

Technologies
Python
PyTorch
Flask
OpenCV
NumPy
Pandas
HTML / CSS / JavaScript
Git & GitHub
Results

The developed system achieved:

Classification Accuracy: 98.24%

Segmentation IoU: 85.12%

Weight Prediction MAE/MAPE: 21.57 kg/13.4%
