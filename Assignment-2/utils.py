import numpy as np
from typing import Callable


class DecisionTree:
    """
    Implements the Decision Tree algorithm for binary classification.
    :attrs:
        criterion: The criterion to use for splitting nodes. Can be "gini" or "entropy".
        max_depth: The maximum depth of the tree.
    The following attributes are available after the model is fit to the data.
        x_train: The training data for the algorithm.
        y_train: The training labels for the algorithm.
        tree: The Decision Tree structure.
    """

    def __init__(self, criterion: str|None = "gini", max_depth: int|None = None):
        if criterion not in ["gini", "entropy"]:
            raise ValueError(f"Unknown criterion '{criterion}'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None


    def impurity(self, y: np.ndarray) -> float:
        """
        Calculate and return the impurity of a set of labels.
        """

        total_samples = len(y)
        if total_samples == 0:
            return 0

        unique_classes, class_counts = np.unique(y, return_counts=True)
        proportions = class_counts / total_samples
        if self.criterion == "gini":
            return 1 - np.sum(proportions**2)
        else:
            return -np.sum(proportions * np.log2(proportions))


    def cost_function(self, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate and return the information gain or impurity based on the chosen criterion
        achieved by a potential split.
        :params:
            y: Array of class labels in the parent node.
            left (array-like): Array of class labels in the left child node.
            right (array-like): Array of class labels in the right child node.
        """
        p = len(left) / (len(left) + len(right))
        parent = self.impurity(y)
        left, right = self.impurity(left), self.impurity(right)
        return parent - (p * left + (1 - p) * right)


    def make_split(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Find the best feature and value to split a node.
        :params:
            x: Input features.
            y: Array of class labels.
        :return:
            dict: {"index": best_feature_index, "value": best_split_value}
        """
        best_gain = -1
        best_split = None

        for feature_index in range(x.shape[1]):
            values = np.unique(x[:, feature_index])
            for value in values:
                L_mask = x[:, feature_index] <= value
                R_mask = x[:, feature_index] > value

                impurity = self.cost_function(y, y[L_mask], y[R_mask])
                L_impurity = self.impurity(y[L_mask])
                R_impurity = self.impurity(y[R_mask])
                gain = impurity - (len(y[L_mask]) / len(y) * L_impurity + len(y[R_mask]) / len(y) * R_impurity)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {"index": feature_index, "value": value}

        return best_split


    def check_max_depth(self, depth: int) -> bool:
        """
        Checks if the maximum depth of the tree has been reached using the tree's current depth.
        """
        return self.max_depth is not None and depth >= self.max_depth


    def fit(self, x_train: np.ndarray, y_train: np.ndarray, depth: int = 0) -> dict:
        """
        Build the Decision Tree recursively.
        :params:
            x_train: Input features.
            y_train: Array of class labels.
            depth: Current depth of the tree (used for max depth).
        """
        if len(np.unique(y_train)) == 1 or self.check_max_depth(depth):
            return {"class": np.bincount(y_train).argmax()}

        best_split = self.make_split(x_train, y_train)
        if best_split is None:
            return {"class": np.bincount(y_train).argmax()}

        L_mask = x_train[:, best_split['index']] <= best_split['value']
        R_mask = x_train[:, best_split['index']] > best_split['value']

        L_tree = self.fit(x_train[L_mask], y_train[L_mask], depth + 1)
        R_tree = self.fit(x_train[R_mask], y_train[R_mask], depth + 1)

        self.tree = {**best_split, "left": L_tree, "right": R_tree}
        return self.tree


    def predict_sample(self, tree: dict[str, int|dict[str, int]], x: np.ndarray) -> int:
        """
        Predict class labels for a single input using the trained Decision Tree.
        """
        if "class" in tree:
            return tree["class"]
        if x[tree["index"]] <= tree["value"]:
            return self.predict_sample(tree["left"], x)
        else:
            return self.predict_sample(tree["right"], x)


    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels of multiple inputs.
        """
        if self.tree is None:
            raise Exception("The model has not been trained yet. Please call fit() first.")
        return np.array([self.predict_sample(self.tree, x) for x in x_test])


    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the accuracy of the model.
        """
        return np.mean(self.predict(x_test) == y_test)