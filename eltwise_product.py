from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import constraints, regularizers, initializers, activations
import tensorflow.keras.backend as K
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

        self.W_constraint = constraints.get(W_constraint)
        self.input_dim = input_dim

        super(EltWiseProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer: create the trainable weights.
        """
        assert len(input_shape) == 2
        self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]

        # Initialize the weights matrix
        self.w = self.add_weight(shape=(input_shape[1][-1] // self.downsampling_factor, input_shape[1][-1] // self.downsampling_factor),
                                 initializer=self.init,
                                 name='kernel',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        # Call the parent class' build method
        super(EltWiseProduct, self).build(input_shape)

    def call(self, inputs):
        """
        Perform the element-wise product.
        """
        # Unpack the inputs
        x, y = inputs

        # Compute the element-wise product
        output = x * y

        return output

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape: should be same as input shape.
        """
        return input_shape[0]

    def get_config(self):
        """
        Return the config of the layer for serialization.
        """
        config = super(EltWiseProduct, self).get_config()
        config.update({
            'downsampling_factor': self.downsampling_factor,
            'init': initializers.serialize(self.init),
            'activation': activations.serialize(self.activation),
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'input_dim': self.input_dim
        })
        return config
