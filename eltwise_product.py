from keras.layers.core import Layer, InputSpec
from keras import constraints, regularizers, initializers, activations
import keras.backend as K
import tensorflow as tf


class EltWiseProduct(Layer):
    """
    Custom Keras layer that performs an element-wise product with a learnable weight matrix (prior).
    This layer upsamples the learned weights and multiplies them with the input feature map.

    Args:
        downsampling_factor (int): Factor by which weights are downsampled relative to input.
        init (str): Initialization method for weights.
        activation (str): Activation function to apply (default linear).
        W_regularizer: Regularizer for weights.
        activity_regularizer: Regularizer for layer output.
        W_constraint: Constraint for weights.
        input_dim (int): Expected input dimension.
    """
    def __init__(self, downsampling_factor=10, init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):

        self.downsampling_factor = downsampling_factor
        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get
