# Image Classification using Convolutional Neural Network (CNN)

This repository contains an image classification project implemented using a Convolutional Neural Network (CNN) in Python. The project demonstrates how deep learning models can automatically learn visual features from images and classify them into predefined categories.

## Overview

Image classification is a fundamental problem in computer vision where an input image is assigned a label.  
In this project, a CNN is used to extract spatial features such as edges, textures, and shapes, and then use those features for classification. The project is implemented using a Jupyter Notebook for ease of experimentation and visualization.

## Technologies Used

Python 3  
TensorFlow / Keras  
NumPy  
Matplotlib  
Jupyter Notebook  

## Dataset

The dataset should be organized using a directory-based structure, where each folder represents a class. Separate directories are used for training, validation, and testing.

Example structure:

data  
  train  
    class_1  
    class_2  
  validation  
    class_1  
    class_2  
  test  
    class_1  
    class_2  

You may use a custom dataset or any publicly available image dataset. Dataset paths can be modified inside the notebook.

## Project Structure

Image-Classification--CNN-  
  CNN.ipynb  
  happytest.jpg  
  sadtest.jpg  
  data  
    train  
    validation  
    test  
  README.md  

CNN.ipynb contains the full workflow including data preprocessing, CNN model creation, training, and evaluation.  
happytest.jpg and sadtest.jpg are sample images used for testing predictions.

## Installation

Clone the repository and move into the project directory.

git clone https://github.com/Sabeer65/Image-Classification--CNN-.git  
cd Image-Classification--CNN-  

Install the required dependencies.

pip install tensorflow numpy matplotlib  

## Usage

Launch Jupyter Notebook and open the main notebook.

jupyter notebook CNN.ipynb  

Run the notebook cells sequentially to load the dataset, preprocess images, train the CNN model, and evaluate performance.

## Model Training

The CNN model includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification. Training parameters such as epochs, batch size, optimizer, and learning rate can be adjusted inside the notebook.

## Testing and Prediction

After training, the model is used to predict unseen images such as happytest.jpg and sadtest.jpg. The predicted class is displayed based on the trained model output.

## Results

The model achieves reasonable accuracy depending on dataset size and quality. Training and validation accuracy and loss values are visualized using plots generated during training.

## Limitations

Model performance depends heavily on dataset size and diversity.  
Small datasets may lead to overfitting.  
The project does not include deployment or real-time inference.

## Future Improvements

Add data augmentation techniques.  
Improve CNN architecture depth.  
Save and load trained model weights.  
Add confusion matrix and classification report.  
Convert the notebook into a deployable application.

## License

This project is licensed under the MIT License.
