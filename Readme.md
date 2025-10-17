# Real-Time Smile Detection using a Fine-Tuned VGG19 Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)

This project implements a complete pipeline for detecting smiles in real-time using deep learning and computer vision. The system leverages the lightweight YuNet for face detection and a fine-tuned VGG19 model for highly accurate smile classification.

### Live Demo
![Smile Detection Demo]

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Workflow](#project-workflow)
- [Setup and Installation](#setup-and-installation)
- [Model Architecture and Performance](#model-architecture-and-performance)
- [Future Work](#future-work)
- [License](#license)

## Overview
The goal of this project was to build an accurate and efficient smile detector. The process begins by isolating faces from a source image using the **YuNet face detector**. These cropped faces are then preprocessed and fed into a **VGG19 Convolutional Neural Network**, which has been fine-tuned on the GENKI-4k dataset to classify whether the person is smiling or not. The final notebook integrates this entire pipeline with a live webcam feed for a real-time demonstration.

## Key Features
- **Accurate Face Detection**: Uses the efficient YuNet model (only 76k parameters) for fast and reliable face extraction.
- **High-Performance Classification**: Employs a fine-tuned VGG19 model, achieving **~94.66% validation accuracy**.
- **Real-Time Processing**: The entire pipeline is optimized to run smoothly on a live webcam feed.
- **End-to-End Project**: Covers the full data science workflow from data preparation and face detection to model training and deployment.

## Technology Stack
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, Scikit-learn, NumPy, Matplotlib, scikit-image
- **Tools**: Jupyter Notebook, Google Colab (for GPU-accelerated training)

## Project Workflow
The project is organized into three Jupyter Notebooks, which are designed to be run sequentially:

1.  **`1.FD_YuNet.ipynb`**: This notebook handles the first step of the pipeline: face detection. It takes the raw images from the GENKI-4k dataset, uses the YuNet model to detect facial bounding boxes, and saves the cropped faces into new directories (`cropped_smile` and `cropped_non_smile`).

2.  **`2.VGG19_SmileDetection_Finetunning.ipynb`**: This is the core machine learning notebook. It loads the preprocessed face images, builds the VGG19 model, and fine-tunes it on our specific task. The final trained model (`.h5` file) is saved at the end of this process.

3.  **`3.VideoCapturing.ipynb`**: This notebook provides a real-time demonstration. It captures video from a webcam, applies the YuNet face detector and the trained smile classifier to each frame, and displays the "Smile" or "Non-smile" prediction live on the screen.

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

**1. Clone the Repository**
```bash
git clone https://github.com/armankhl/smile-detection-project.git
cd smile-detection-project
```

**2. Create and Activate a Virtual Environment** (Recommended)
```bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the Dataset and Pre-trained Model**
-   **Dataset**: Download the **GENKI-4k Dataset** from [Kaggle](https://www.kaggle.com/datasets/mansorour/genki4k). Unzip the file and place the `smile` and `non_smile` folders inside a `Dataset/` directory in the project's root folder.
-   **Trained Smile Model**: Download the `smile_detection_model_VGG_HF17_v6.h5` file from the [**Releases**](https://github.com/armankhl/smile-detection-project/releases) page of this repository and place it in the project's root directory.

## Model Architecture and Performance

The classification model is based on the **VGG19** architecture, pre-trained on ImageNet.

-   **Fine-Tuning Strategy**: To adapt the model for our task, the first 17 layers of the VGG19 base were frozen. A custom classification head was added, consisting of Global Average Pooling, Batch Normalization, Dropout, and two Dense layers.
-   **Total Parameters**: 20.4 million
-   **Trainable Parameters**: 9.8 million

### Performance
The model was trained for 10 epochs and achieved the following results on the validation set:
-   **Validation Loss**: `0.179`
-   **Validation Accuracy**: `94.66%`

#### Training History
| Model Accuracy | Model Loss |
| :---: | :---: |
| ![Accuracy Plot](https://github.com/armankhl/smile-detection-project/blob/main/demo/accuracy_plot.png?raw=true) | ![Loss Plot](https://github.com/armankhl/smile-detection-project/blob/main/demo/loss_plot.png?raw=true) |

## Future Work
- **Code Refactoring**: Convert helper functions into a `utils.py` file to keep notebooks cleaner.
- **Model Exploration**: Experiment with other architectures like ResNet50 or InceptionV3 to compare performance.
- **Web Application**: Build a simple web interface using Streamlit or Flask to allow users to upload images or use their webcam for detection without running notebooks.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.