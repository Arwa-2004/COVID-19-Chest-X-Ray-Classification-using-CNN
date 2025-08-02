# COVID-19-Chest-X-Ray-Classification-using-CNN
<img width="945" height="408" alt="image" src="https://github.com/user-attachments/assets/4382f53d-d496-42e7-90b6-6b648d05ac83" />

## Motivation 
Early and accurate diagnosis using chest X-ray images can help in timely treatment and containment. This project aims to classify chest X-rays into COVID-19 positive or normal cases using a deep learning model. A TensorFlow/Keras implementation of a Convolutional Neural Network (CNN) is used to classify chest X-ray images into **COVID-19** or **Normal** categories.

## Dataset
Dataset from Kaggle [COVID-19 Chest X-Ray Detection dataset](https://www.kaggle.com/datasets/akhilkasare/covid19-chest-xray-detection/data). A total of 672 files in which the dataset is organized in directories:
  - `Train/covid` and `Train/non_covid` for training images
  - `Test/covid` and `Test/non_covid` for validation images
- Images are grayscale and resized to 500x500 pixels.

## Model Architecture
- Input: 500x500 grayscale images
- 5 convolutional layers with ReLU activations and max-pooling
- Dropout (0.5) for regularization
- Fully connected layers: 512 → 256 → 128 → 64 neurons
- Output layer with 2 neurons and softmax activation (binary classification)

## Training Details
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 16
- Number of epochs: 13
- Data augmentation applied on training images (rescaling, shear, zoom, horizontal flip)


## Results
| Dataset  | Accuracy (%) | Loss   |
| -------- | ------------ | ------ |
| Training | **94.04%**   | 0.1579 |
| Testing  | **89.04%**   | 0.5531 |


## Dependencies

![NumPy](https://img.shields.io/badge/NumPy-1.24.3-blue)
![pandas](https://img.shields.io/badge/pandas-2.0.3-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.14.0-yellow)



