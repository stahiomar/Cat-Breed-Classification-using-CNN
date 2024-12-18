# Cat Breed Classification using CNN

This project uses Convolutional Neural Networks (CNN) to classify images of three types of cats: Siamese, Orange, and Sphynx. The model is trained to recognize these breeds based on image data and allows for real-time classification of new images.

## Tools Used

- **TensorFlow 2.x**: A deep learning framework used to build, train, and evaluate the CNN model.
- **Keras**: A high-level neural networks API integrated with TensorFlow, used for defining the model architecture.
- **NumPy**: Used for numerical operations, especially for handling image arrays.
- **Matplotlib**: A plotting library used to display images and predictions.
- **Python 3.x**: Programming language used for the project.

## Purpose

The purpose of this project is to create an image classification model that can automatically identify the breed of a cat from images. Specifically, the model is trained to classify images of the following cat breeds:
- **Siamese**
- **Orange**
- **Sphynx**

The model is designed to be easily retrained with additional images or adapted for other classification tasks.

## Data Split: Train, Validation, and Test

In order to properly evaluate and fine-tune the model, the dataset is separated into three distinct subsets: **Training**, **Validation**, and **Test**. Here's the purpose of each:

### 1. **Training Set**
- **Purpose**: The training set is used to train the model. It contains the majority of the images and allows the model to learn the patterns and features that define each cat breed.
- **Size**: Typically, 70-80% of the total dataset is used for training.

### 2. **Validation Set**
- **Purpose**: The validation set is used to evaluate the model during training, allowing us to fine-tune the model's hyperparameters. This set helps in monitoring the model's performance and making adjustments to avoid overfitting.
- **Size**: Typically, 10-15% of the total dataset is used for validation.

### 3. **Test Set**
- **Purpose**: The test set is used to evaluate the final performance of the trained model. It is never used during the training or validation process. This set is crucial to check how well the model generalizes to new, unseen data.
- **Size**: Typically, 10-15% of the total dataset is used for testing.

## How It Works

1. **Data Preprocessing**: Images are resized to 150x150 pixels and normalized to a range of 0 to 1 to prepare them for model input.
2. **Model Architecture**: The model consists of several convolutional layers followed by max-pooling layers to extract features, and dense layers to make predictions. The model uses `softmax` activation for the final layer, making it suitable for multi-class classification.
3. **Training**: The model is trained using the `categorical_crossentropy` loss function and the `adam` optimizer.
4. **Prediction**: After training, the model can predict the breed of a cat from a new image.
