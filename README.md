# AI-Powered Fruit Recognition System

## Overview

The AI-Powered Fruit Recognition System is a deep learning application that classifies fruit images using Transfer Learning with MobileNetV2.

The model is trained on a multi-class fruit dataset and can identify different fruit categories from uploaded images through a Streamlit web application.

## Features

* Image Classification using Deep Learning
* MobileNetV2 Transfer Learning
* Real-Time Prediction
* Confidence Score Display
* Streamlit Web Interface

## Technologies Used

* Python
* TensorFlow
* MobileNetV2
* NumPy
* Streamlit
* Pillow

## Dataset

The model was trained on a fruit image dataset containing 10 fruit categories:

* Apple
* Avocado
* Banana
* Cherry
* Kiwi
* Mango
* Orange
* Pineapple
* Strawberry
* Watermelon

## Model Architecture

* MobileNetV2 (Pretrained on ImageNet)
* Global Average Pooling Layer
* Dropout Layer
* Dense Output Layer

## Results

The trained model achieved high classification accuracy on the validation dataset and successfully predicts fruit categories from uploaded images.

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Future Improvements

* Support additional fruit categories
* Deploy using Streamlit Cloud
* Add Top-3 Prediction Visualization
* Mobile-Friendly Interface
