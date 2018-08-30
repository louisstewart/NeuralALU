import tensorflow as tf
from tensorflow.python.layers import base as base_layer


class NeuralALU(base_layer.Layer):

    def __init__(self, outputs, name=None, kernel_initializer=None):
        super(NeuralALU, self).__init__(name=name)
        self._outputs = outputs
        self._k_init = kernel_initializer

    def compute_output_shape(self, input_shape):
        return self._outputs

    def build(self, inputs_shape):
        """
        Build required weights matrices.
        Args:
            inputs_shape: size of weights.

        Returns:
            build cell
        """
        size = (inputs_shape[-1], self._outputs)
        self._w = self.add_variable("w_hat", shape=size, initializer=self._k_init, trainable=True)
        self._m = self.add_variable("m_hat", shape=size, initializer=self._k_init, trainable=True)
        self._g = self.add_variable("g_hat", shape=size, initializer=self._k_init, trainable=True)
        self._epsilon = 0.0000001

        self.built = True

    def call(self, inputs, **kwargs):
        """
        Perform one step of NALU
        Args:
            inputs: 2D Tensor, [batch, feature_size]
            **kwargs:

        Returns:
            2D Tensor of NALU activations
        """
        W = tf.multiply(tf.tanh(self._w, name='tanh_w_hat'), tf.sigmoid(self._m, name="sigmoid_m_hat"), name="W")
        G = tf.sigmoid(tf.matmul(inputs, self._g), name="G")
        m = tf.exp(tf.matmul(tf.log(tf.add(tf.abs(inputs), self._epsilon)), W))
        a = tf.matmul(inputs, W, name="nac_activation")

        y = tf.add(tf.multiply(G, a), tf.multiply((1-G), m))

        return y