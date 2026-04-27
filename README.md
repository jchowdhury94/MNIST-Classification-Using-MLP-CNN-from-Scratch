# MNIST Classification Project (CSC4851/6851)

## Overview

This project implements two deep learning models from scratch (without using deep learning libraries such as torch.nn or TensorFlow):

* Task 1: Multi-Layer Perceptron (MLP)
* Task 2: Convolutional Neural Network (CNN)

Both models are trained and evaluated on the MNIST dataset for handwritten digit classification (0–9).

---

## Environment Requirements

Make sure the following are installed:

* Python 3.x
* NumPy
* PyTorch
* torchvision

Install dependencies using:

```
pip install numpy torch torchvision
```

---

## Dataset

The MNIST dataset is used for training and testing.

* Training samples: 60,000
* Testing samples: 10,000
* Image size: 28 × 28 (grayscale)

The dataset will be **automatically downloaded** when running the code for the first time. No manual download is required.

---

## Project Structure

```
Jannati_Chowdhury_Source_Code/
│
├── Jannati_Chowdhury_Task1.py   # MLP implementation
├── Jannati_Chowdhury_Task2.py   # CNN implementation
├── read_MNIST.py                # Provided dataloader
├── README.md                   # Project instructions
```

---

## How to Run

### Task 1: MLP

Run the following command:

```
python Jannati_Chowdhury_Task1.py
```

This will:

* Load the MNIST dataset
* Train the MLP model for 30 epochs
* Print training loss per epoch
* Output final test accuracy

---

### Task 2: CNN

Run the following command:

```
python Jannati_Chowdhury_Task2.py
```

This will:

* Load the MNIST dataset
* Train the CNN model for 5 epochs
* Print training loss during training
* Output final test accuracy

---

## Model Details

### MLP (Task 1)

* Input size: 784 (flattened 28×28 image)
* Hidden layer: **256 neurons**, Sigmoid activation
* Output layer: 10 neurons with Softmax activation
* Loss function: Cross-Entropy Loss
* Training: Mini-batch gradient descent (batch size = 128, 30 epochs)

---

### CNN (Task 2)

* Convolutional layer: 1 filter, kernel size = 5×5 (valid convolution)
* Activation: ReLU
* Flatten layer
* Fully connected layer: 10 neurons
* Output layer: Softmax activation
* Loss function: Cross-Entropy Loss
* Training: Mini-batch gradient descent (batch size = 128, 5 epochs)

---

## Implementation Notes

* All components (forward propagation, backward propagation, activation functions, and loss functions) are manually implemented using NumPy.
* No deep learning libraries (e.g., torch.nn, TensorFlow, Keras) are used for model construction.
* PyTorch is only used for loading and preprocessing the dataset.

---

## Author

Jannati Chowdhury
