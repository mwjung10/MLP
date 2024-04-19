import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
#from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MLPClassifier import MLPClassifier


#Load the digits dataset
digits = datasets.load_digits(as_frame=True)

# #Display digit
# plt.figure(1, figsize=(3, 3))
# plt.imshow(digits.images[70], cmap='binary')
# plt.show()

# print(type(digits))
# print(digits.keys())
# n_samples, n_features = digits.data.shape
# print((n_samples, n_features))
# X = digits.data
# y = digits.target
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#mlp = MLPClassifier(neurons_size=40)
mlp = MLPClassifier(hidden_layer_sizes=(40),  # number of neurons in the hidden layer
                    activation='logistic',   # activation function (logistic sigmoid)
                    alpha=1e-4,     # helps preventing overfittting
                    solver='sgd',    # optimization algorithm
                    tol=1e-4,       # tolerance for optimzation; determines when to stop it
                    random_state=1,  #  This parameter sets the random seed for reproducibility. By setting it to 1, the results will be reproducible across multiple runs.
                    learning_rate_init=.1, # learning rate for the optimization algorithm. It determines the size of the steps
                    verbose=True)  # print progress messages to the console during training. Setting it to True enables verbose output.
mlp.train(X_train, y_train)
preds = mlp.predict(X_test)

print(accuracy_score(preds, y_test))

# import json

# data = datasets.load_digits(as_frame=True)
# json_str = data.frame.to_json(orient='records')

# with open('digits_dataset.json', 'w') as f:
#     f.write(json_str)