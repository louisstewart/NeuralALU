import tensorflow as tf
from tensorflow.python.layers import base as base_layer


class NeuralAccumulator(base_layer.Layer):

    def __init__(self):
        super(NeuralAccumulator, self).__init__()
        pass

    def build(self, inputs_shape):
        """
        Build the NAC cell by creating the necessary weights and accumulator matrices
        Args:
            inputs_shape: shape of cell

        Returns:
            builds cell weights
        """
        self._w = self.add_variable("w_hat", shape=inputs_shape)
        self._m = self.add_variable("m_hat", shape=inputs_shape)

        self.built = True

    def call(self, inputs, **kwargs):
        """
        Run one step of Neural Accumulator
        Args:
            inputs: input tensor, 2D, [batch, input_size]

        Returns:
            2D Tensor
        """

        W = tf.multiply(tf.tanh(self._w, name='tanh_w_hat'), tf.sigmoid(self._m, name="sigmoid_m_hat"), name="W")
        a = tf.matmul(inputs, W, name="nac_activation")
        return a

