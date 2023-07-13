# Image Classification on CIFAR-10 Dataset

This project implements an image classification task on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) model built with PyTorch. 

## Files and Directories

### 1. `utils.py`

This file includes the data transformation functions using the `albumentations` library. It also includes a `visualise_transformation` function that lets you visualize transformations applied on images.

### 2. `train_test.py`

This file includes the training and testing loop functions. 

### 3. `model.py`

This file includes the definition of the Convolutional Neural Network (CNN) model. 

### 4. `assignment_9.ipynb`

This file includes the main training file where the model is trained for a defined number of epochs. 

## Data Augmentation

Data augmentation is performed using the `albumentations` library. The following augmentations are used:

- PadIfNeeded: Pads the image if its size is less than the required size.
- CoarseDropout: Drops out rectangular regions in the image and the dropped regions are set to a defined value.
- CenterCrop: Crops the central part of the image.
- Affine: Applies a random affine transformation.
- Normalize: Normalizes the image.
- ToTensorV2: Converts the image to a PyTorch tensor.

## Model Architecture

The model architecture includes convolutional layers, dropout layers, ReLU activations, batch normalization layers, and a global average pooling layer.

## Training and Testing

The model is trained for a defined number of epochs using the SGD optimizer with momentum and the OneCycleLR learning rate scheduler. The training and testing accuracy and loss are recorded and plotted.
![image](https://github.com/hemant1456/ERA_Course/assets/19394814/721850b0-ffc9-4177-9c93-785363f44dfa)


## Visualization

The transformations applied on the images are visualized using `matplotlib`.
<img width="760" alt="Screenshot 2023-06-30 at 12 25 57 PM" src="https://github.com/hemant1456/ERA_Course/assets/19394814/e545d4d4-d2be-4b76-a161-a74a97404b4c">



## Performance Analysis

The training and testing accuracy and loss are plotted against the number of epochs to analyze the performance of the model.

![image](https://github.com/hemant1456/ERA_Course/assets/19394814/5837dc06-b4bf-4ee2-b39d-f8d778bd9074)

