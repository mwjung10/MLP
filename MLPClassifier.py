import numpy as np
from sklearn import datasets
import random


class MLPClassifier:
    def init(self, neurons_size):
        self.num_layers = len(neurons_size)
        self.neurons_size = neurons_size
        self.biases = np.array(np.random.randn(y, 1) for y in neurons_size[1:])  #
        self.weights = [np.random.randn(y, x) for x,y in zip(neurons_size[:-1], neurons_size[1:])]

    def fit(self):
        pass

    def sigmoid(self, x):
        return 1./(1.+np.exp(-x))

    def feedForward(self, input):
        # Feeds forward array "input" of inputs to next layer unitl it reaches output layer

        for bias, weights in zip(self.biases, self.weights):
            input = self.sigmoid(np.dot(weights, input) + bias)  # np.dot is the dot product of two arrays (matrices)

        return input

    def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001,
                        stopping_threshold = 1e-6):

    #stochastic gradient descent works by picking out a randomly chosen mini-batch of training inputs

    # def update_mini_batch(self, mini_batch, eta):
    #     # updates biases and weights when you apply gradient descent using backpropagation

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]






if __name__ == "__main__":
    mlp = MLPClassifier(10 , 15, 20)