from __future__ import division
from keras.models import Model
from keras.layers import Dropout, Activation, Input, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.applications import VGG16
import keras.backend as K
import math

from eltwise_product import EltWiseProduct
from config import *


def ml_net_model(img_rows=480, img_cols=640, downsampling_factor_net=8, downsampling_factor_product=10):
    """
    Builds the ML-Net model with a pretrained VGG16 backbone and custom prior learning layer.
    
    Args:
        img_rows (int): Input image height.
        img_cols (int): Input image width.
        downsampling_factor_net (int): Downsampling factor in feature extractor (VGG16).
        downsampling_factor_product (int): Downsampling factor in prior learning layer.

    Returns:
        Keras Model: ML-Net saliency prediction model.
    """
    # Input tensor with shape (height, width, channels)
    input_ml_net = Input(shape=(img_rows, img_cols, 3))

    # Load VGG16 pretrained on ImageNet without top fully connected layers
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_ml_net)

    # Extract outputs of intermediate conv and pooling layers for multi-scale feature combination
    conv1_2 = base_model.get_layer('block1_conv2').output
    conv1_pool = base_model.get_layer('block1_pool').output
    conv2_2 = base_model.get_layer('block2_conv2').output
    conv2_pool = base_model.get_layer('block2_pool').output
    conv3_3 = base_model.get_layer('block3_conv3').output
    conv3_pool = base_model.get_layer('block3_pool').output
    conv4_3 = base_model.get_layer('block4_conv3').output
    conv4_pool = base_model.get_layer('block4_pool').output
    conv5_3 = base_model.get_layer('block5_conv3').output

    # Modify last pooling layer (block4_pool) to have stride=1 to match original paper's architecture
    conv4_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv4_3)

    # Concatenate multi-scale feature maps from conv3_pool, conv4_pool, and conv5_3 layers
    concatenated = Concatenate(axis=-1)([conv3_pool, conv4_pool, conv5_3])

    # Apply dropout for regularization
    dropout = Dropout(0.5)(concatenated)

    # Intermediate convolution to reduce dimensionality and learn encoding
    int_conv = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal')(dropout)

    # Final 1x1 convolution to produce single-channel saliency map
    pre_final_conv = Conv2D(1, (1, 1), activation='relu', kernel_initializer='glorot_normal')(int_conv)

    # Calculate prior learning weights size based on downsampling factors
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product

    # Custom prior learning layer with L2 regularization, initialized to zeros
    eltprod = EltWiseProduct(kernel_initializer='zeros', kernel_regularizer=l2(1 / (rows_elt * cols_elt)))(pre_final_conv)

    # Apply ReLU activation to final output
    output_ml_net = Activation('relu')(eltprod)

    # Define the model with inputs and outputs
    model = Model(inputs=input_ml_net, outputs=output_ml_net)

    # Print input/output shapes of layers (ignore layers without shape info)
    for layer in model.layers:
        try:
            print(layer.input_shape, layer.output_shape)
        except Exception:
            continue

    return model


def loss(y_true, y_pred):
    """
    Custom loss function that normalizes predictions and calculates weighted MSE.
    
    Args:
        y_true: Ground truth saliency maps.
        y_pred: Predicted saliency maps.
    
    Returns:
        Tensor: Computed loss value.
    """
    # Find max prediction per example for normalization
    max_y = K.max(K.max(y_pred, axis=1, keepdims=True), axis=2, keepdims=True)
    max_y = K.repeat_elements(max_y, shape_r_gt, axis=1)
    max_y = K.repeat_elements(max_y, shape_c_gt, axis=2)
    
    # Weighted mean squared error loss, prioritizing low saliency regions
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
