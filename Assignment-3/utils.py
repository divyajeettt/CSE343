import numpy as np


class NeuralNetwork:
    """
    Implements the Neural Network for classification.
    :attrs:
        N: The number of layers in the network.
        hidden_layer_sizes: An array of size N-2 integers defining the number of neurons in each layer.
        alpha: The learning rate.
        epochs: The number of iterations to run gradient descent.
        batch_size: The batch size to use for gradient descent.
        activation: The activation function to use. Can be "sigmoid", "tanh", "ReLU", "LeakyReLU", "linear", or "softmax".
        weight_init: The weight initialization function to use. Can be "random", "zero", or "normal".

    The following attributes are available after the model is fit to the data.
        x_train: The training data for the algorithm.
        y_train: The training labels for the algorithm.
        weights: The weights for the neural network.

    The following extra attributes are available for the model evaluation.
        train_loss: The loss of the model on the training data per epoch.
        valid_loss: The loss of the model on the validation data per epoch.
        train_accuracy: The accuracy of the model on the training data per epoch.
        valid_accuracy: The accuracy of the model on the validation data per epoch.
    """

    N: int
    hidden_layer_sizes: np.ndarray|list[int]
    alpha: float|None
    epochs: int|None
    batch_size: int|None
    activation: str|None
    weight_init: str|None

    def __init__(self, N, hidden_layer_sizes, alpha=0.01, activation="sigmoid", epochs=50, batch_size=128, weight_init="normal"):
        self.N = N
        assert len(hidden_layer_sizes) == self.N-2, "Number of hidden layers must be equal to N-2."

        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size

        self.activation = activation.casefold()
        if self.activation not in ["sigmoid", "tanh", "arctan", "relu", "leakyrelu", "linear"]:
            raise ValueError("Activation function must be either 'sigmoid', 'tanh', 'arctan', 'ReLU', 'LeakyReLU', or 'linear'.")

        self.weight_init = weight_init.casefold()
        if self.weight_init not in ["random", "zero", "normal"]:
            raise ValueError("Weight initialization function must be either 'random', 'zero', or 'normal'.")

        self.weights = None
        self.train_loss = np.full(self.epochs, np.nan)
        self.valid_loss = np.full(self.epochs, np.nan)
        self.train_accuracy = np.zeros(self.epochs)
        self.valid_accuracy = np.zeros(self.epochs)


    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of the input.
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input
        """
        return {
            "sigmoid": 1 / (1 + np.exp(-x)),
            "tanh": np.tanh(x),
            "arctan": np.arctan(x),
            "relu": np.maximum(0, x),
            "leakyrelu": np.maximum(0.01*x, x),
            "linear": x
        }[self.activation]


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the activation function to the input
        """
        return {
            "sigmoid": x * (1 - x),
            "tanh": 1 - np.tanh(x)**2,
            "arctan": 1 / (1 + x**2),
            "relu": np.where(x > 0, 1, 0),
            "leakyrelu": np.where(x > 0, 1, 0.01),
            "linear": np.ones_like(x)
        }[self.activation]


    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss of the model on the data.
        """
        return np.mean((y - self.predict_proba(x, intercept=False))**2)


    def _initialize_weights(self) -> None:
        """
        Initialize the weights for the neural network.
        """
        if self.weight_init == "random":
            initializer = np.random.random
        elif self.weight_init == "zero":
            initializer = lambda size: np.zeros(shape=size)
        else:
            initializer = np.random.normal

        self.weights = [initializer(size=(self.hidden_layer_sizes[0], self.x_train.shape[1]))]
        for i in range(1, self.N-2):
            self.weights.append(initializer(size=(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1])))
        self.weights.append(initializer(size=(self.classes, self.hidden_layer_sizes[-1])))


    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> None:
        """
        Fit the data to the model.
        """
        self.classes = np.unique(y_train).size
        self.x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        self.y_train = np.eye(self.classes)[y_train]
        self.x_valid = np.hstack((np.ones((x_valid.shape[0], 1)), x_valid))
        self.y_valid = np.eye(self.classes)[y_valid]
        self._initialize_weights()


    def propagate(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Perform one iteration of forward and backward propagation on the data.
        """
        self.forward_propagate(x)
        self.backward_propagate(y)
        self.update_weights()


    def forward_propagate(self, x: np.ndarray) -> None:
        """
        Perform forward propagation on the data.
        """
        self.activations = [x]
        for i in range(self.N-1):
            self.activations.append(self.activate(self.activations[-1] @ self.weights[i].T))


    def backward_propagate(self, y: np.ndarray) -> None:
        """
        Perform backward propagation on the data.
        """
        self.deltas = [y - self.activations[-1]]
        for i in range(self.N-2, 0, -1):
            self.deltas.append(self.deltas[-1] @ self.weights[i] * self.gradient(self.activations[i]))
        self.deltas.reverse()


    def update_weights(self) -> None:
        """
        Update the weights of the neural network.
        """
        for i in range(self.N-1):
            self.weights[i] += self.alpha * (self.activations[i].T @ self.deltas[i]).T


    def train(self, verbose: bool|None = False) -> None:
        """
        Train the model.
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before training.")

        for epoch in range(self.epochs):
            indices = np.random.permutation(self.x_train.shape[0])
            x_train = self.x_train[indices]
            y_train = self.y_train[indices]
            for i in range(0, self.x_train.shape[0], self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]
                self.propagate(x_batch, y_batch)
            self.train_loss[epoch] = self.loss(self.x_train, self.y_train)
            self.valid_loss[epoch] = self.loss(self.x_valid, self.y_valid)
            self.train_accuracy[epoch] = self.score(self.x_train, np.argmax(self.y_train, axis=1), intercept=False)
            self.valid_accuracy[epoch] = self.score(self.x_valid, np.argmax(self.y_valid, axis=1), intercept=False)
            if verbose:
                print(f"Epoch {str(epoch + 1).zfill(3)}, Train Loss: {self.train_loss[epoch]:.5f}, Valid Loss: {self.valid_loss[epoch]:.5f}", end="\r")


    def predict(self, x: np.ndarray, intercept: bool|None = True) -> np.ndarray:
        """
        Predict the labels for the input data.
        """
        return np.argmax(self.predict_proba(x, intercept), axis=1)


    def predict_proba(self, x: np.ndarray, intercept: bool|None = True) -> np.ndarray:
        """
        Predict the probabilities for the input data.
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before predicting.")
        else:
            x = np.hstack((np.ones((x.shape[0], 1)), x)) if intercept else x
            self.forward_propagate(x)
            return NeuralNetwork.softmax(self.activations[-1])


    def score(self, x: np.ndarray, y: np.ndarray, intercept: bool|None = True) -> float:
        """
        Compute the accuracy of the model on the data.
        """
        return np.mean(self.predict(x, intercept) == y)