# NOTE: This file is not used when loading preprocessed .npy files.
from __future__ import division
import cv2
import numpy as np


#-----------------------#
#this file could be changed based on the preprocess on img 
#-----------------------#


# ------------------------------------------------------------------------
# Function: padding
# Purpose: Resize an image or saliency map to a fixed size (shape_r x shape_c)
#          while preserving its aspect ratio by padding the borders with black.
# Inputs:
#   - img: Input image or saliency map (RGB or grayscale)
#   - shape_r, shape_c: Target height and width
#   - channels: 3 for RGB images, 1 for grayscale maps
# Output:
#   - Padded image of shape (shape_r, shape_c)
# ------------------------------------------------------------------------
def padding(img, shape_r=480, shape_c=640, channels=3):
    # Create a black canvas with the target shape
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    # Get original dimensions
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    # Resize image while maintaining aspect ratio
    if rows_rate > cols_rate:
        # Resize by height
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        # Center horizontally
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        # Resize by width
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        # Center vertically
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

# ------------------------------------------------------------------------
# Function: preprocess_images
# Purpose: Preprocess input RGB images before feeding into ML-Net model.
# Steps:
#   1. Load image.
#   2. Pad to fixed size (480x640).
#   3. Subtract ImageNet mean (standard VGG preprocessing).
#   4. Transpose dimensions to (batch, channels, height, width).
# Inputs:
#   - paths: List of file paths to input images
#   - shape_r, shape_c: Desired shape (default is 480x640)
# Output:
#   - Preprocessed image batch as NumPy array
# ------------------------------------------------------------------------
def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))  # (N, H, W, C)

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    # Subtract ImageNet mean (in BGR order)
    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    # Transpose to NCHW format for Keras with Theano/TF 1.x backend
    ims = ims.transpose((0, 3, 1, 2))  # (N, C, H, W)

    return ims

# ------------------------------------------------------------------------
# Function: preprocess_maps
# Purpose: Preprocess ground truth saliency maps.
# Steps:
#   1. Load grayscale map.
#   2. Pad to fixed shape.
#   3. Normalize pixel values to range [0, 1].
# Inputs:
#   - paths: List of file paths to saliency maps
#   - shape_r, shape_c: Target shape
# Output:
#   - Normalized map batch as NumPy array (N, 1, H, W)
# ------------------------------------------------------------------------
def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)  # load as grayscale
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, 0] = padded_map.astype(np.float32) / 255.0  # normalize to [0, 1]

    return ims

# ------------------------------------------------------------------------
# Function: postprocess_predictions
# Purpose: Resize predicted saliency map back to original shape for visualization.
# Steps:
#   1. Resize keeping aspect ratio.
#   2. Center-crop if needed.
#   3. Normalize output to range [0, 255].
# Inputs:
#   - pred: Predicted map (2D NumPy array)
#   - shape_r, shape_c: Final output shape (typically 480x640)
# Output:
#   - Processed prediction map, ready for display or saving
# ------------------------------------------------------------------------
def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    # Resize and center-crop to fit desired shape
    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255  # Normalize to 0–255 range
