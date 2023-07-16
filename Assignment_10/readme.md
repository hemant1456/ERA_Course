# Image Classification on CIFAR-10 Dataset

This project implements an image classification task on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) model built with PyTorch. 

## Files and Directories

### 1. `utils.py`

This file includes the data transformation functions using the `albumentations` library. It also includes a `visualise_transformation` function that lets you visualize transformations applied on images.

### 2. `train_test.py`

This file includes the training and testing loop functions. 

### 3. `model.py`

This file includes the definition of the Convolutional Neural Network (CNN) model. 

### 4. `assignment_10.ipynb`

This file includes the main training file where the model is trained for a defined number of epochs. 

## Data Augmentation

Data augmentation is performed using the `albumentations` library. The following augmentations are used:

- PadIfNeeded: Pads the image to 40x40 size.
- RandomCrop: to get the size 32x32, helps in regularisation and better test accuracy
- CutOut- helps in regularisation and better test accuracy
- Normalize: Normalizes the image.
- ToTensorV2: Converts the image to a PyTorch tensor.

## Model Architecture

The model architecture includes convolutional layers, ReLU activations, batch normalization layers
The model tries to replicate the "Resnet architecture"

## Training and Testing

The model is trained for 24 epochs using the Adam optimizer and the OneCycleLR learning rate scheduler. The training and testing accuracy and loss are recorded and plotted.
![image](https://github.com/hemant1456/ERA_Course/assets/19394814/39bbdc60-016d-4b64-8fd1-bc3e38486e63)



## Visualization

The transformations applied on the images are visualized using `matplotlib`.
![image](https://github.com/hemant1456/ERA_Course/assets/19394814/13707e9e-d118-4711-9965-4d8a347477f6)




## Performance Analysis

The below image show some of the misclassified imagess by the model

![image](https://github.com/hemant1456/ERA_Course/assets/19394814/8fefb414-e88e-42c4-bec0-aef777ea4760)


