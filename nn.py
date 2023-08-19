import math

import numpy as np

from activation import Sigmoid, RELU, Activation, ActivationCache


class NeuralNetwork:
    def __init__(self, layer_dims, activations, initialization='naive'):
        """
        Initializes Neural Network
        :param layer_dims: python array containing the dimensions of each layer in our network
        :param activations: activation functions of the layers
        :param initialization: initialization strategy:
                               naive   - W[i] ~ N(0, 1)*0.1
                               xavier  - W[i] ~ U (-(sqrt(6)/sqrt(n + m)), sqrt(6)/sqrt(n + m))  where n us the number
                                               of inputs to the node and m is the number of outputs from the layer
                               he      - W[i] ~ N (0, sqrt(2/n)) , where n is the number of inputs to the node.
                               smart   - he for RELU and xavier for SIGMOID
        """

        self.W = []  # weights
        self.b = []  # biases
        L = len(layer_dims)

        self.activations: list[Activation] = []
        if len(activations) > 0:
            if isinstance(layer_dims[0], int):
                assert (len(activations) == L - 1)
            else:
                assert (len(activations) == L)
            for x in activations:
                if isinstance(x, str):
                    if x == 'sigmoid':
                        self.activations.append(Sigmoid())
                    elif x == 'relu':
                        self.activations.append(RELU())
                    else:
                        raise Exception('Unknown activation [' + x + ']')
                elif isinstance(x, Activation):
                    self.activations.append(x)
                else:
                    raise Exception('activation should be of type str or Activation')
        else:
            self.activations = [Sigmoid() for _ in range(L)]
        if isinstance(layer_dims[0], int):
            for d in range(1, L):
                init_f = None
                if initialization == 'naive':
                    init_f = NeuralNetwork.__init_naive
                elif initialization == 'xavier':
                    init_f = NeuralNetwork.__init_xavier
                elif initialization == 'he':
                    init_f = NeuralNetwork.__init_he
                elif initialization == 'smart':
                    if isinstance(self.activations[d - 1], RELU):
                        init_f = NeuralNetwork.__init_he
                    else:
                        init_f = NeuralNetwork.__init_xavier
                if init_f is None:
                    raise Exception(f'unknown initialization {initialization}')
                self.W.append(init_f(layer_dims[d], layer_dims[d - 1]))
                self.b.append(np.zeros((layer_dims[d], 1)))
        else:
            for tpl in layer_dims:
                self.W.append(tpl[0])
                self.b.append(tpl[1])

    # Init methods
    @staticmethod
    def __init_naive(n, m):
        return np.random.randn(n, m) * 0.01

    @staticmethod
    def __init_xavier(n, m):
        d = np.sqrt(6.0)
        lower, upper = -(d / np.sqrt(n + m)), (d / np.sqrt(n + m))
        return np.random.randn(n, m) * (upper - lower)

    @staticmethod
    def __init_he(n, m):
        return np.random.randn(n, m) * np.sqrt(2 / m)

    "Performs forward propagation through the network given an input X. It returns the output of " \
    "the network and a list of caches containing intermediate values needed for backpropagation."
    def forward(self, X):
        caches = []
        A = X
        n = len(self.W)
        for i in range(0, n):
            A_prev = A
            W = self.W[i]
            b = self.b[i]
            act = self.activations[i]
            A, cache = act.forward(A_prev, W, b)
            caches.append(cache)

        return A, caches

    def backward(self, AL, Y, caches: list[ActivationCache], regularization_factor: float = 0):
        """
        :param regularization_factor: regularization hyperparameter, scalar
        :param AL: probability vector, output of the forward propagation
        :param Y: true "label" vector (containing 0 if non-cat, 1 if cat)
        :param caches: list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[i], for i in range(L-1) i.e i = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        :return: Array with the gradients [(dW, db)]

        """
        # grads = {}
        grads = []
        L = len(caches)  # the number of layers
        current_cache = caches[L - 1]

        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = AL - Y

        act = self.activations[L - 1]
        beta = regularization_factor / m
        dA, dW, dB = act.backward(dAL, current_cache)
        if beta > 0:
            dW += current_cache.linear_cache.W * beta
        grads.append((dW, dB))
        for i in reversed(range(L - 1)):
            current_cache = caches[i]
            act = self.activations[i]
            dA, dW, dB = act.backward(dA, current_cache)
            if beta > 0:
                dW += current_cache.linear_cache.W * beta
            grads.append((dW, dB))
        return list(reversed(grads))

    "Updates the weights and biases of the network using the gradients and learning rate."
    def update_parameters(self, grads, learning_rate):
        i = 0
        for w, b, g in zip(self.W, self.b, grads):
            dW, dB = g
            self.W[i] = w - learning_rate * dW
            self.b[i] = b - learning_rate * dB
            i += 1

    def __update_parameters_momentum(self, grads, velocity, beta, learning_rate):
        i = 0
        for w, b, g, v in zip(self.W, self.b, grads, velocity):
            dW, dB = g
            vW, vB = v
            next_vW = beta * vW + (1 - beta) * dW
            next_vB = beta * vB + (1 - beta) * dB
            self.W[i] = w - learning_rate * next_vW
            self.b[i] = b - learning_rate * next_vB
            velocity[i] = (next_vW, next_vB)
            i += 1

    "Calculates the cross-entropy loss between the predicted values (AL) and the true labels (Y)."
    @staticmethod
    def cross_entropy(AL, Y):
        m = Y.shape[1]
        epsilon = 0.00001
        AL[AL == 0] = epsilon
        AL[AL == 1.0] = 1-epsilon
        cost = np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL))) / -m
        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())
        return cost

    "Calculates the total cost, including the cross-entropy loss and an optional regularization term"
    def cost(self, AL, Y, regularization_factor: float = 0.0):
        ce = NeuralNetwork.cross_entropy(AL, Y)
        m = Y.shape[1]
        beta = regularization_factor / (2 * m)
        if regularization_factor > 0:
            l2 = 0
            for w in self.W:
                l2 = l2 + np.sum(np.square(w))
            return ce + l2 * beta
        else:
            return ce

    @staticmethod
    def __random_mini_batches(x, y, mini_batch_size=64, seed=0):

        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        if mini_batch_size == 0:
            return [(x, y)]

        np.random.seed(seed)
        m = x.shape[1]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = x[:, permutation]
        shuffled_Y = y[:, permutation]  # .reshape((1, m))
        # shuffled_X = np.random.permutation(x)
        # shuffled_Y = np.random.permutation(y)

        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            end = m - mini_batch_size * math.floor(m / mini_batch_size)
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def __initialize_momentum(self):
        velocity = []
        for w, b in zip(self.W, self.b):
            velocity.append((np.zeros_like(w),
                             np.zeros_like(b)))
        return velocity

    " Trains the neural network using gradient descent or momentum optimization. "
    "It takes the input data (x) and labels (y), the number of epochs, learning rate, "
    "mini-batch size, optimizer type, momentum parameter, "
    "and other optional parameters. It returns the costs and epoch numbers if specified."
    def fit(self, x: np.array, y: np.array,
            epochs: int,
            learning_rate: float,
            mini_batch_size: int = 64,
            optimizer: str = "gd",
            beta=0.9,  # for momentum
            return_costs: bool = False,
            print_cost: bool = False,
            report_every: int = 10):
        """

        :param x: input
        :param y: labels
        :param epochs: number of learning epochs
        :param learning_rate:
        :param mini_batch_size: 0 for ordinal gradient descent, >0 for stochastic gradient descent
        :param optimizer: gd/momentum https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
        :param beta: parameter for momentum optimizer
        :param return_costs: if True, returns array of costs for each 100-th invocation
        :param print_cost: be verbose, if True

        """
        velocity = []

        if optimizer == 'momentum':
            velocity = self.__initialize_momentum()

        costs = []
        epoch_nums = []
        seed = 10

        for i in range(epochs):
            seed = seed + 1
            minibatches = NeuralNetwork.__random_mini_batches(x, y, mini_batch_size, seed)

            batch_cost = []
            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch

                yh, cache = self.forward(minibatch_x)
                cost = self.cost(yh, minibatch_y)
                batch_cost.append(cost)
                grads = self.backward(yh, minibatch_y, cache)
                if len(velocity) == 0:
                    self.update_parameters(grads, learning_rate)
                else:
                    self.__update_parameters_momentum(grads, velocity, beta, learning_rate)
            cost = np.average(batch_cost)
            if return_costs and i % report_every == 0:
                costs.append(cost)
                epoch_nums.append(i)
            if print_cost and i % report_every == 0:
                print(f'epoch = {i}, cost = {cost}')
        return costs, epoch_nums

    def predict(self, x):
        yhat, cache = self.forward(x)
        return np.argmax(yhat, axis=0)

    def score(self, x, y):
        yhat = self.predict(x)
        return NeuralNetwork.accuracy(yhat, y)

    @staticmethod
    def accuracy(yhat, ytrue):
        N = len(ytrue)
        return np.sum(yhat == ytrue) / N


