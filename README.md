# Real-Time Sign Language Recognition

A Computer Vision system that translates American Sign Language (ASL) alphabets into text in real-time using a standard webcam. This project bridges communication gaps by leveraging low-latency Deep Learning models.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-green)

## Features
* **Real-Time Inference:** Optimized pipeline using **OpenCV** to process video frames with sub-second latency.
* **Robust Detection:** Utilizes **Google MediaPipe** for skeletal hand tracking, ensuring accuracy even in varying lighting conditions.
* **Custom CNN Architecture:** Trained a Convolutional Neural Network on extracted geometric feature vectors for high-efficiency classification.

## Project Structure
* `inference.py`: The main script that launches the webcam and runs real-time prediction.
* `dl-project.ipynb`: The Jupyter Notebook used for data preprocessing, model training, and evaluation.
* `model.keras`: Actual model file.

## How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yyouretoast/real-time-sign-language-recognition.git](https://github.com/yyouretoast/real-time-sign-language-recognition.git)
