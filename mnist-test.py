import sys
import time
import joblib
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from nn import NeuralNetwork

test_counter = 1

"The run_nn function is defined to run a test with the neural network. "
"It takes various parameters such as training and testing data, "
"layer dimensions, activation functions, initialization method, number of epochs, "
"learning rate, mini-batch size, optimizer, and beta value. "
"It creates an instance of the NeuralNetwork class with the specified parameters and fits the "
"network to the training data. It then computes the accuracy on the testing data and plots the "
"cost function over epochs."
def run_nn(
        x_train,
        y_train,
        x_test,
        y_test,
        y_real,
        layer_dims,
        activations,
        initialization,
        epochs: int,
        learning_rate,
        mini_batch_size,
        optimizer,
        beta):
    global test_counter
    print(f'------------------ TEST {test_counter} ----------------')
    test_counter += 1
    network = NeuralNetwork(layer_dims, activations, initialization=initialization)
    costs, epoch_nums = network.fit(x_train, y_train,
                                    epochs=epochs,
                                    optimizer=optimizer,
                                    learning_rate=learning_rate,
                                    mini_batch_size=mini_batch_size,
                                    beta=beta,
                                    print_cost=True,
                                    return_costs=True,
                                    report_every=2)
    plt.clf()
    plt.figure(figsize=(10, 6))

    accuracy = network.score(x_test, y_real)
    print(f'Accuracy = {accuracy}')
    plt.title(f'Cost function over epochs. Total accuracy = {accuracy}')
    plt.xlabel("Epochs x 100")
    plt.ylabel("Cost")
    plt.plot(costs, epoch_nums)

    plt.show()

" load the MNIST dataset."
def load_data():
    X = np.load('MNIST-data.npy')
    Y = np.load("MNIST-lables.npy")
    X = X.reshape(-1, 28 * 28)
    return X, Y

" loads and preprocesses the dataset"
def load_and_process_data():
    # Loads and preprocesses the dataset
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype('float32').values
    y = mnist.target.astype('int')
    """
    X = np.load('MNIST-data.npy')
    Y = np.load("MNIST-lables.npy")
    X = X.reshape(-1, 28 * 28)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    y_encoder = OneHotEncoder(sparse_output=False)
    y_encoder.fit(Y.reshape(-1, 1))
    x_encoder = StandardScaler()
    x_encoder.fit(X)

    x_test = x_encoder.transform(X_test).T
    x_train = x_encoder.transform(X_train).T
    y_test = y_encoder.transform(Y_test.reshape(-1, 1)).T
    y_train = y_encoder.transform(Y_train.reshape(-1, 1)).T
    return x_train, y_train, x_test, y_test, Y_test


def run_tests():
    np.seterr(all="ignore")
    x_train, y_train, x_test, y_test, y_test_real = load_and_process_data()
    num_of_features = x_train.shape[0]
    num_of_classes = len(np.unique(y_test_real))
    np.random.seed(1)

    # TEST 1
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 70, num_of_classes],
           ['relu', 'sigmoid'],
           initialization='smart',
           epochs=30,
           optimizer='gd',
           learning_rate=0.1,
           mini_batch_size=9,
           beta=0.9
           )

    # TEST 2
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 70, num_of_classes],
           [],
           initialization='smart',
           epochs=15,
           optimizer='gd',
           learning_rate=0.3,
           mini_batch_size=9,
           beta=0.6
           )

    # TEST 3
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 50, 30, 50, num_of_classes],
           ['relu', 'sigmoid', 'relu', 'sigmoid'],
           initialization='smart',
           epochs=30,
           optimizer='momentum',
           learning_rate=0.025,
           mini_batch_size=64,
           beta=0.7)

    # TEST 4
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 30, num_of_classes],
           ['relu', 'sigmoid'],
           initialization='smart',
           epochs=30,
           optimizer='momentum',
           learning_rate=0.025,
           mini_batch_size=40,
           beta=0.9)

    # TEST 5
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 30, num_of_classes],
           ['relu', 'sigmoid'],
           initialization='smart',
           epochs=30,
           optimizer='momentum',
           learning_rate=0.03,
           mini_batch_size=50,
           beta=0.9)

    # TEST 6
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 30, num_of_classes],
           ['relu', 'sigmoid'],
           initialization='smart',
           epochs=10,
           optimizer='gd',
           learning_rate=0.08,
           mini_batch_size=64,
           beta=0.9)

    # TEST 7
    run_nn(x_train, y_train, x_test, y_test, y_test_real,
           [num_of_features, 30, num_of_classes],
           ['relu', 'sigmoid'],
           initialization='smart',
           epochs=30,
           optimizer='gd',
           learning_rate=0.17,
           mini_batch_size=40,
           beta=0.9)


if __name__ == '__main__':
    run_tests()
