import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, layer_dims):
        """ 
        Initialize your neural network. Then use fit() for training the network. And predict() to predict a value after training.

        Parameters: 
        layer_dims (`list` of `int`): Layer dimension of the network. e.g: [2, 2, 1] for 2 input nodes, 1 output node and 1 hidden layer with 2 nodes.
        """
        self.parameters = {}
        self.L = len(layer_dims)
        self.costs = []

        for l in range(1, self.L):
            self.parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self, Z):
        A = np.maximum(0, Z)
        cache = Z

        return A, cache

    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

        return dZ

    def linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)

        if activation == "relu":
            A, activation_cache = self.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def predict(self, X, inside_class=False):
        """ 
        Predict a value using trained neural network.

        Parameters: 
        X (`numpy array`): Dimension of shape should be (no_of_input_nodes, no_of_test_data)

        Returns: 
        `numpy array`: Dimension is (no_of_output_nodes, no_of_test_data)
        """
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters["W" + str(l)], self.parameters["b" + str(l)], "relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, self.parameters["W" + str(L)], self.parameters["b" + str(L)], "sigmoid")
        caches.append(cache)

        if inside_class:
            return AL, caches
        else:
            return AL

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        logprobs = (Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL))
        cost = - np.sum(logprobs) / m
        cost = np.squeeze(cost)

        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache                        # Cache from linear_forward function
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(cache[1].T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        # Cache from linear_activation_forward function
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        if activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def model_backward(self, AL, Y, caches):
        grads = {}
        # Caches from predict function
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[-1]                  # Cacher for the last layer
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")

        return grads

    def update_parameters(self, grads, learning_rate):

        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - grads["dW" + str(l+1)] * learning_rate
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - grads["db" + str(l+1)] * learning_rate

    def fit(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=100, show_cost=True):
        """
        Train you neural network.

        Parameters: 

        X (`numpy array`): Dimension of shape should be (no_of_input_nodes, no_of_test_data)

        Y (`numpy array`): Dimension of shape should be (no_of_output_nodes, no_of_test_data)

        learning_rate (`float`): Learning rate of the network

        num_iterations (`int`): Number of iterations

        print_cost (`int`): -1 for not printing cost and print_cost>=1 for printing after every print_cost iterations

        show_cost (`boolean`): To determine if it should show the costs of every 100 iterations on graph.
        """
        for i in range(num_iterations):
            AL, caches = self.predict(X, True)
            cost = self.compute_cost(AL, Y)
            grads = self.model_backward(AL, Y, caches)
            self.update_parameters(grads, learning_rate)

            if print_cost != -1 and i % print_cost == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if i % 100 == 0:
                self.costs.append(cost)

        if show_cost:
            plt.plot(np.squeeze(self.costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
