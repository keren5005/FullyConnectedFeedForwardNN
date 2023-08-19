import numpy as np

# Required for computing the backward pass efficiently
"This class represents the cache used in the "
"linear forward pass. It stores the input A, weight matrix W, and bias vector b"
class LinearForwardCache:
    def __init__(self, A, W, b):
        self.A = A
        self.W = W
        self.b = b

"his class represents the cache used in the activation function. "
"It stores the output Z of the activation function and the linear cache "
class ActivationCache:
    def __init__(self, Z: np.array, linear_cache: LinearForwardCache):
        self.activation_cache = Z
        self.linear_cache = linear_cache

"This is the base class for activation functions. " \
"It defines the forward and backward methods for computing the activation and its derivatives, respectively."
class Activation:
    def __init__(self):
        pass

    def forward(self, A, W, b) -> (np.array, ActivationCache):
        z = Activation.linear_forward(A, W, b)
        return self._forward_impl(z), ActivationCache(z, LinearForwardCache(A, W, b))

    def backward(self, dA, cache: ActivationCache) -> (np.array, np.array, np.array):
        z = cache.activation_cache
        dZ = self._backward_impl(dA, z)
        dA, dW, db = Activation.linear_backward(dZ, cache.linear_cache)
        return dA, dW, db

    def _forward_impl(self, z) -> np.array:
        return np.array([0])

    def _backward_impl(self, dA, z) -> np.array:
        return np.array([0])

    @staticmethod
    # Implements linear part of the activation function
    def linear_forward(A, W, b) -> np.array:
        z = np.dot(W, A) + b
        assert (z.shape == (W.shape[0], A.shape[1]))
        return z

    @staticmethod
    def linear_backward(dZ: np.array, cache: LinearForwardCache):
        """
            Implement the linear portion of backward propagation for a single layer (layer l)

            Arguments:
            dZ -- Gradient of the cost with respect to the linear output (of current layer l)
            cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

            Returns:
            dA -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as cache.A
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        m = cache.A.shape[1]
        dW = np.dot(dZ, cache.A.T) / m

        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(cache.W.T, dZ)

        assert (dA.shape == cache.A.shape)
        assert (dW.shape == cache.W.shape)
        assert (db.shape == cache.b.shape)
        return dA, dW, db

"This class extends Activation and implements the sigmoid activation function. "
"It overrides the _forward_impl and _backward_impl methods to compute the sigmoid and its derivative."
class Sigmoid(Activation):
    def __init__(self):
        super(Activation, self).__init__()

    def _forward_impl(self, z) -> np.array:
        return 1 / (1 + np.exp(-z))

    def _backward_impl(self, dA, z) -> np.array:
        s = 1 / (1 + np.exp(-z))
        dZ = dA * s * (1 - s)
        assert (dZ.shape == z.shape)
        return dZ

"This class extends Activation and implements the rectified linear unit (ReLU) activation function. "
"It overrides the _forward_impl and _backward_impl methods to compute the ReLU and its derivative."
class RELU(Activation):
    def __init__(self):
        super(Activation, self).__init__()

    def _forward_impl(self, z) -> np.array:
        a = np.maximum(0, z)
        assert (a.shape == z.shape)
        return a

    def _backward_impl(self, dA, z) -> np.array:
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well.
        dZ[z <= 0] = 0
        assert (dZ.shape == z.shape)
        return dZ
