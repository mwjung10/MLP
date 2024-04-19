from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from MLPClassifier import MLPClassifier

# Load the digits dataset
digits = load_digits()

# Preprocess the data
X = digits.data
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate your MLPClassifier
# mlp = MLPClassifier(neurons_size=[X_train.shape[1], 30, 10])  # Input layer size is the number of features, output layer size is the number of classes (10 digits)
mlp = MLPClassifier([8*8, 40, 10])
mlp.train([X_train, y_train], 10000, learning_rate=0.01, test_data=[X_test, y_test])
# Train the classifier
# mlp.train(training_data=list(zip(X_train_scaled, y_train)), epochs=100, learning_rate=0.1, stopping_threshold=1e-5, test_data=list(zip(X_test_scaled, y_test)))
