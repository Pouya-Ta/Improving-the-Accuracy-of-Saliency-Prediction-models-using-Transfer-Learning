import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Activation, Input, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.layers import UpSampling2D
import tensorflow.keras.backend as K

from eltwise_product import EltWiseProduct
from config import *

def ml_net_model(img_rows=224, img_cols=224, downsampling_factor_net=8, downsampling_factor_product=10, freeze_vgg=True):
    input_ml_net = Input(shape=(img_rows, img_cols, 3))

    # Load VGG16 with or without trainable weights
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_ml_net)
    for layer in base_model.layers:
        layer.trainable = not freeze_vgg  # Freeze or unfreeze based on parameter

    conv1_2 = base_model.get_layer('block1_conv2').output
    conv1_pool = base_model.get_layer('block1_pool').output
    conv2_2 = base_model.get_layer('block2_conv2').output
    conv2_pool = base_model.get_layer('block2_pool').output
    conv3_3 = base_model.get_layer('block3_conv3').output
    conv3_pool = base_model.get_layer('block3_pool').output
    conv4_3 = base_model.get_layer('block4_conv3').output
    conv4_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv4_3)
    conv5_3 = base_model.get_layer('block5_conv3').output

    conv5_3_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5_3)
    # Encoding
    concatenated = Concatenate(axis=-1)([conv3_pool, conv4_pool, conv5_3_upsampled])
    dropout = Dropout(0.5)(concatenated)
    int_conv = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal')(dropout)
    pre_final_conv = Conv2D(1, (1, 1), activation='relu', kernel_initializer='glorot_normal')(int_conv)

    # Prior
    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zeros', W_regularizer=l2(1 / (rows_elt * cols_elt)))(pre_final_conv)
    output_ml_net = Activation('relu')(eltprod)

    return Model(inputs=input_ml_net, outputs=output_ml_net)


def loss(y_true, y_pred):
    # Prevent NaN from max=0
    max_y = K.maximum(K.max(K.max(y_pred, axis=1, keepdims=True), axis=2, keepdims=True), K.epsilon())
    
    # Expand max_y to full map size
    max_y = K.repeat_elements(max_y, shape_r_gt, axis=1)
    max_y = K.repeat_elements(max_y, shape_c_gt, axis=2)
    
    # Clip y_true to avoid zero division
    y_true_clipped = K.clip(y_true, 0.0, 1.0)
    
    return K.mean(K.square((y_pred / max_y) - y_true) / (1.0 - y_true + 0.1))