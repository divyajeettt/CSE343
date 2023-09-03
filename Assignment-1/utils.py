import numpy as np
from typing import Callable


def split_dataset(x: np.ndarray, y: np.ndarray, split: list[float]) -> tuple[tuple[np.ndarray]]:
    """
    split the dataset into train, test, and validation sets.
    :params:
        x: The input data.
        y: The labels for the input data.
        split: The ratio of the dataset to be used for training, testing, and validation respectively.
    """

    if not np.isclose(np.sum(split), 1, atol=1e-3):
        raise ValueError("Sum of splits must be 1.")

    # Shuffle the dataset
    indices = np.random.permutation(x.shape[0])
    x = x[indices]
    y = y[indices]

    x_train = x[:int(split[0] * x.shape[0])]
    y_train = y[:int(split[0] * y.shape[0])]
    x_valid = x[int(split[0] * x.shape[0]):int((split[0] + split[1]) * x.shape[0])]
    y_valid = y[int(split[0] * y.shape[0]):int((split[0] + split[1]) * y.shape[0])]
    x_tests = x[int((split[0] + split[1]) * x.shape[0]):]
    y_tests = y[int((split[0] + split[1]) * y.shape[0]):]

    return (x_train, y_train), (x_tests, y_tests), (x_valid, y_valid)


def standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize the input data.
    """
    return (x - np.mean(x)) / np.std(x)


class LogisticRegression:
    """
    Implements the Logistic Regression algorithm for binary classification.
    :attrs:
        alpha: The learning rate.
        epochs: The number of iterations to run gradient descent.
        batch_size: The batch size to use for gradient descent.
        regularization: The regularization function to use. Can be "L1", "L2", or None.
        lamda: The regularization parameter. Must be specified if regularization is not None.
        logistic: The logistic function to use. Can be "sigmoid" or "tanh".

    The following attributes are available after the model is fit to the data.
        x_train: The training data for the algorithm.
        y_train: The training labels for the algorithm.
        weights: The weights for the logistic regression model (initialized to 0).

    The following extra attributes are available for the model evaluation.
        train_loss: The loss of the model on the training data per iteration.
        valid_loss: The loss of the model on the validation data per iteration.
        train_accuracy: The accuracy of the model on the training data per iteration.
        valid_accuracy: The accuracy of the model on the validation data per iteration.
    """

    alpha: float|None
    epochs: int|None
    batch_size: int|None
    lamda: float|None
    logistic: str|None

    def __init__(self, alpha=0.01, epochs=100, batch_size=1, regularization=None, lamda=None, logistic="sigmoid"):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None

        self.logistic = logistic
        if self.logistic not in ["sigmoid", "tanh"]:
            raise ValueError("Logistic function must be either 'sigmoid' or 'tanh'.")

        self.regularization = regularization
        self.lamda = 0.0 if lamda is None else lamda
        if self.regularization not in ["L1", "L2", None]:
            raise ValueError("Regularization function must be either 'L1', 'L2', or None.")
        elif self.regularization is not None and self.lamda == 0.0:
            raise ValueError("lamda must be specified if regularization is not None.")

        self.train_loss = np.full(self.epochs, np.nan)
        self.valid_loss = np.full(self.epochs, np.nan)
        self.train_accuracy = np.zeros(self.epochs)
        self.valid_accuracy = np.zeros(self.epochs)


    def __repr__(self) -> str:
        """
        Returns the string representation of the model.
        """
        return f"LogisticRegression(alpha={self.alpha}, epochs={self.epochs})"


    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input.
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before activation.")
        elif self.logistic == "sigmoid":
            return 1 / (1 + np.exp(-x @ self.weights))
        else:
            return (1 + np.tanh(x @ self.weights)) / 2


    def regularize(self) -> tuple[float]:
        """
        Compute the regularization term for the model. Returns terms for loss and gradient.
        """
        if self.regularization is None:
            return 0.0, 0.0
        elif self.regularization == "L1":
            return self.lamda * np.linalg.norm(self.weights, ord=1), self.lamda * np.sign(self.weights)
        else:
            return self.lamda * np.linalg.norm(self.weights, ord=2)**2, 2 * self.lamda * self.weights


    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss of the model.
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before computing loss.")
        else:
            hypothesis = self.activate(x)
            return -np.mean(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)) + self.regularize()[0]


    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the model.
        """
        hypothesis = self.activate(x)
        if self.logistic == "sigmoid":
            return (x.T * (hypothesis - y)).sum(axis=1) + self.regularize()[1]
        else:
            return 0.5 * (x.T * (hypothesis - y) * (1 - hypothesis**2)).sum(axis=1) + self.regularize()[1]


    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> None:
        """
        Fit the model to the data.
        """
        self.x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
        self.y_train = y_train
        self.x_valid = np.hstack((np.ones((x_valid.shape[0], 1)), x_valid))
        self.y_valid = y_valid
        self.weights = np.random.rand(self.x_train.shape[1])


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
                self.weights = self.weights - self.alpha * self.gradient(x_batch, y_batch)
            self.train_loss[epoch] = self.loss(self.x_train, self.y_train)
            self.valid_loss[epoch] = self.loss(self.x_valid, self.y_valid)
            self.train_accuracy[epoch] = self.accuracy(self.x_train, self.y_train, intercept=False)
            self.valid_accuracy[epoch] = self.accuracy(self.x_valid, self.y_valid, intercept=False)
            if verbose:
                print("Epoch:", str(epoch + 1).zfill(4), end="\r")


    def predict(self, x: np.ndarray, intercept: bool|None = True) -> np.ndarray:
        """
        Predict the labels for the input data.
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before prediction.")
        else:
            x = np.hstack((np.ones((x.shape[0], 1)), x)) if intercept else x
            return (self.activate(x) >= 0.5).astype(int)


    def accuracy(self, x: np.ndarray, y: np.ndarray, intercept: bool|None = True) -> float:
        """
        Compute the accuracy of the model.
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before computing accuracy.")
        else:
            return np.mean(self.predict(x, intercept) == y)


    def confusion_matrix(self, x_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Compute the confusion matrix for the model. Returns matrix in the order
        [[TP, FP],
         [FN, TN]]
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before computing confusion matrix.")

        y_pred = self.predict(x_test)
        return np.array([
            [np.sum(y_test &   y_pred), np.sum(1-y_test &  y_pred)],
            [np.sum(y_test & 1-y_pred), np.sum(1-y_test & 1-y_pred)]
        ])


    def confusion_metrics(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float]:
        """
        Compute the confusion metrics for the model. Returns in the order
        [accuracy, precision, recall, f1].
        """
        if self.weights is None:
            raise ValueError("Model must be fit to data before computing confusion metrics.")

        TP, FP, FN, TN = self.confusion_matrix(x_test, y_test).flatten()
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1