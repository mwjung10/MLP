import numpy as np
import sklearn


class MLPClassifier:
    def init(self, hidden_layers):
        self.num_layers = len(hidden_layers)
        self.layer_sizes = hidden_layers
        self.biases = np.array(np.random.randn(y, 1) for y in hidden_layers)
        # self.weights = np.array([],dtype=float)

    def fit(self):
        pass