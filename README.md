# Fully Connected Feed Forward Neural Network

This repository contains Python code for building a fully connected feed forward neural network. The code provides implementations for key components of a neural network, including linear layers and activation functions such as Sigmoid and ReLU. This neural network architecture is widely used in machine learning applications like image recognition and natural language processing.

## Table of Contents

- [Introduction](#introduction)
- [Key Components](#key-components)
- [Activation Functions](#activation-functions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

A feed-forward neural network, also known as a fully connected neural network, is a type of artificial neural network where information flows in one direction, from the input layer to the output layer. It consists of layers of interconnected neurons, allowing it to process complex data and make predictions or classifications. This repository provides code for implementing a fully connected feed forward neural network.

## Key Components

The code in this repository provides implementations for the following key components:

- `LinearForwardCache`: This class represents the cache used in the linear forward pass. It stores the input `A`, weight matrix `W`, and bias vector `b`.

- `ActivationCache`: This class represents the cache used in the activation function. It stores the output `Z` of the activation function and the linear cache.

- `Activation`: This is the base class for activation functions. It defines the forward and backward methods for computing the activation and its derivatives, respectively.

- `Sigmoid`: This class extends `Activation` and implements the sigmoid activation function.

- `RELU`: This class extends `Activation` and implements the rectified linear unit (ReLU) activation function.

## Activation Functions

This repository provides implementations for two popular activation functions:

- **Sigmoid**: The sigmoid activation function maps input values to the range (0, 1). It is commonly used in binary classification tasks.

- **RELU**: The rectified linear unit (ReLU) activation function outputs the input as-is if it is positive, and outputs zero for negative inputs. It is widely used in deep learning architectures.

## Usage

To use the code in this repository, follow these steps:

1. Clone this repository to your local machine:

```sh
git clone https://github.com/keren5005/fully-connected-feed-forward-nn.git
```

2. Navigate to the repository's directory:

```sh
cd fully-connected-nn
```


3. You can now import and use the provided classes for building and experimenting with a fully connected feed forward neural network.

## Contributing

Contributions to this repository are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or additional features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.