# Satellite Image Semantic Segmentation

A deep learning project for semantic segmentation of satellite imagery using a U-Net architecture implemented with TensorFlow and Keras.

## Project Overview

The goal of this project is to classify every pixel in a satellite image into one of six segmentation classes.

The project uses satellite images along with corresponding color-coded ground-truth masks.

## Dataset

https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery

The dataset is organized into 8 tiles, with 9 image-mask pairs per tile, giving a total of 72 original image-mask pairs.

The segmentation masks contain six different classes represented using RGB colors.

The dataset itself is not included in this repository because of its size and dataset availability/licensing considerations.

## Methodology

The overall pipeline is:

Satellite Image
↓
RGB Mask Processing
↓
RGB to Class-ID Conversion
↓
Train/Test Split
↓
256 × 256 Image Patches
↓
Image Normalization
↓
One-Hot Encoding of Masks
↓
TensorFlow Data Pipeline
↓
U-Net
↓
Pixel-wise Class Prediction
↓
Segmentation Mask

## Model

The project uses a U-Net architecture consisting of:

* Encoder
* Bottleneck
* Decoder
* Skip connections
* Transposed convolution for upsampling
* Six-channel softmax output

The final output provides six class probabilities for every pixel.

## Loss Function

The model uses a combination of:

* Dice Loss
* Categorical Focal Loss

Dice loss focuses on segmentation overlap, while focal loss gives more emphasis to difficult pixel classifications.

## Evaluation

Jaccard / Intersection over Union (IoU) is used as a segmentation metric.

The project also visualizes:

* Input satellite images
* Ground-truth segmentation masks
* Predicted segmentation masks
* Intermediate convolutional activation maps
* Gradient-based heatmaps

## Memory Optimization

Because satellite images can generate many 256 × 256 patches, the project uses a Python generator and TensorFlow `tf.data.Dataset` pipeline instead of loading all patches into memory simultaneously.

This reduces memory usage during training.

## Technologies

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

## Future Improvements

Possible improvements include:

* Data augmentation
* Class-weighted loss
* Per-class IoU and mean IoU
* Tile-level train/validation/test splitting
* Larger or pretrained U-Net encoders
* Better handling of class imbalance
* More extensive hyperparameter tuning

## How to Run

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Download/place the dataset in the configured Google Drive directory.
4. Update the dataset path in the notebook.
5. Run the notebook cells sequentially.
6. Train the U-Net model.
7. Visualize the predicted segmentation masks and heatmaps.

