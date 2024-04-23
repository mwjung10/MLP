import numpy as np
from sklearn import datasets
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


class MLPClassifier:
    def __init__(self, neurons_size):
        self.num_layers = len(neurons_size)
        self.neurons_size = neurons_size
        self.biases = [np.random.randn(y) for y in neurons_size[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(neurons_size[:-1], neurons_size[1:])
        ]
        print([b.shape for b in self.biases])
        print("abc")

    def sigmoid(self, x):
        """
        Returns sigmoid function value for given x
        """
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        """
        Returns sigmoid prime function value for given x
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        """
        Returns ReLU function value for given x
        """
        return x * (x > 0)

    def relu_prime(self, x):
        """
        Returns ReLU function value for given x
        """
        return 1.0 * (x > 0)

    def feedForward(self, input, activation_function):
        # Feeds forward array "input" of inputs to next layer unitl it reaches output layer
        if activation_function.upper() == "SIGMOID":
            for bias, weights in zip(self.biases, self.weights):
                input = self.sigmoid(
                    np.dot(weights, input) + bias
                )  # np.dot is the dot product of two arrays (matrices)
        elif activation_function.upper() == "RELU":
            for bias, weights in zip(self.biases, self.weights):
                input = self.relu(np.dot(weights, input) + bias)
        else:
            raise "invalid activation function"
        return input

    def train(
        self,
        training_data,
        epochs=1000,
        learning_rate=0.0001,
        stopping_threshold=1e-6,
        test_data=None,
        activation_function="SIGMOID",
    ):
        for epoch in range(epochs):
            if learning_rate <= stopping_threshold:
                break
            self.GD(training_data, learning_rate, activation_function)
            if test_data:
                n_test = len(test_data[0])
                print(
                    "Epoch {0}: {1} / {2}".format(
                        epoch, self.evaluate(test_data, activation_function), n_test
                    )
                )
            else:
                print("Epoch {0} complete".format(epoch))

    def GD(self, training_data, learning_rate, activation_function):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        # x - Input
        # y - Desired output

        for x, y in zip(training_data[0], training_data[1]):
            delta_gradient_b, delta_gradient_w = self.backprop(
                x, y, activation_function
            )
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

        self.weights = [
            w - (learning_rate / len(training_data[0])) * gw
            for w, gw in zip(self.weights, gradient_w)
        ]
        self.biases = [
            b - (learning_rate / len(training_data[0])) * nb
            for b, nb in zip(self.biases, gradient_b)
        ]

    def backprop(self, x, y, activation_function="SIGMOID"):
        """
        Returns calculated delta for weights and biases
        """
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer z
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            if activation_function == "SIGMOID":
                activation = self.sigmoid(z)
            elif activation_function == "RELU":
                activation = self.relu(z)
            activations.append(activation)

        if activation_function == "SIGMOID":
            delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(
                zs[-1]
            )

        elif activation_function == "RELU":
            delta = self.cost_derivative(activations[-1], y) * self.relu_prime(zs[-1])

        gradient_b[-1] = delta
        gradient_w[-1] = [activations[-2] * x for x in delta]

        for l in range(2, self.num_layers):
            z = zs[-l]

            if activation_function == "SIGMOID":
                sp = self.sigmoid_prime(z)
            elif activation_function == "RELU":
                sp = self.relu_prime(z)

            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradient_b[-l] = delta
            gradient_w[-l] = [activations[-l - 1] * d for d in delta]
            
        return (gradient_b, gradient_w)

    def evaluate(self, test_data, activation_function="SIGMOID"):
        test_results = [
            (np.argmax(self.feedForward(x, activation_function)), y)
            for (x, y) in zip(test_data[0], test_data[1])
            ]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (2 * (output_activations - y)) 
        #return output_activations - y


if __name__ == "__main__":
    # Load the digits dataset
    digits = load_digits()

    # Preprocess the data
    X = digits.data
    Y = digits.target

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    # Instantiate your MLPClassifier
    mlp = MLPClassifier([8 * 8, 128, 10])
    test_data = [X_test, Y_test]
    mlp.train([X_train, Y_train], 10000, learning_rate=0.01, test_data=test_data, activation_function="SIGMOID")
