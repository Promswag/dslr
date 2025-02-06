import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
# Adding plot_adam_path method to LogisticRegression class
from mpl_toolkits.mplot3d import Axes3D


def preprocessing(df: pd.DataFrame, target_name: str, keep_na: bool=False) -> pd.DataFrame:
    data = df.select_dtypes(include='number')
    classes = {c: i for i, c in enumerate(df[target_name].unique())}
    data[target_name] = df[target_name].map(classes)

    if not keep_na:
        data.dropna(inplace=True)
    else:
        means = data.groupby(target_name).mean()
        data = data.apply(lambda x: x.fillna(means.loc[x[target_name]]), axis=1)
        data[target_name] = data[target_name].astype(int)

    return data


def standardize(lst: pd.Series) -> pd.Series:
    mean = lst.mean()
    std = lst.std()
    return (lst - mean) / std


class LogisticRegression:
    def __init__(self, features: pd.DataFrame, target: pd.Series, learning_rate: float = 0.01, epochs: int = 1000, beta1: float = 0.9, beta2: float = 0.999, batch_size: int = 32):
        if features.shape[0] != target.shape[0]:
            raise ValueError("LogisticRegression: Features and Target must be of the same shape.")

        self.m, self.n_features = features.shape
        self.n_classes = len(target.unique())
        self.features = features
        self.target = target
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.losses = []
        self.weight_history = []
        self.bias_history = []
        self.bias = np.random.randn(self.n_classes)
        self.W = np.random.randn(self.n_classes, self.n_features)

    def gradient_descent(self):
        for _ in range(self.epochs):
            logits = np.dot(self.features, self.W.T) + self.bias
            pred = self.softmax(logits)
            loss_pred = pred.copy()
            pred[range(self.m), self.target] -= 1
            self.W -= self.learning_rate * (np.dot(pred.T, self.features) / self.m)
            self.bias -= self.learning_rate * (np.sum(pred, axis=0) / self.m)
            self.losses.append(self.compute_loss(loss_pred))

    def compute_loss(self, pred):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)
        return -np.mean(np.log(pred[np.arange(self.m), self.target]))

    def compute_loss_adam(self, pred, y_batch):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)

        if np.max(y_batch) >= pred.shape[1]:
            raise ValueError(f"Target index out of range: max(target)={np.max(y_batch)}, but pred.shape={pred.shape}")

        return -np.mean(np.log(pred[np.arange(len(y_batch)), y_batch]))

    def mini_batch_gd(self):
        for _ in range(self.epochs):
            indices = np.random.permutation(self.m)
            for i in range(0, self.m, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch, y_batch = self.features.iloc[batch_indices], self.target.iloc[batch_indices]
                logits = np.dot(X_batch, self.W.T) + self.bias
                pred = self.softmax(logits)
                pred[range(len(y_batch)), y_batch] -= 1
                self.W -= self.learning_rate * (np.dot(pred.T, X_batch) / len(y_batch))
                self.bias -= self.learning_rate * (np.sum(pred, axis=0) / len(y_batch))

    def adam(self):
        epsilon = 1e-8
        m_w, v_w = np.zeros_like(self.W), np.zeros_like(self.W)
        m_b, v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
        t = 0

        for _ in range(self.epochs):
            indices = np.random.permutation(self.m)
            for i in range(0, self.m, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch, y_batch = self.features.iloc[batch_indices], self.target.iloc[batch_indices]
                logits = np.dot(X_batch, self.W.T) + self.bias
                probs = self.softmax(logits)
                probs[range(len(y_batch)), y_batch] -= 1
                grad_w = np.dot(probs.T, X_batch) / len(y_batch)
                grad_b = np.sum(probs, axis=0) / len(y_batch)

                t += 1
                m_w = self.beta1 * m_w + (1 - self.beta1) * grad_w
                v_w = self.beta2 * v_w + (1 - self.beta2) * (grad_w ** 2)
                m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
                v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)

                m_w_hat = m_w / (1 - self.beta1 ** t)
                v_w_hat = v_w / (1 - self.beta2 ** t)
                m_b_hat = m_b / (1 - self.beta1 ** t)
                v_b_hat = v_b / (1 - self.beta2 ** t)

                self.W -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

                self.weight_history.append(self.W.copy())
                self.bias_history.append(self.bias)
                self.losses.append(self.compute_loss_adam(probs, y_batch))   

    def plot_adam_path(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        losses = np.array(self.losses)
        weights = np.array([w.flatten() for w in self.weight_history])
        biases = np.array(self.bias_history)

        ax.plot(weights[:, 0], biases[:, 0], losses, marker='o', linestyle='-', color='b')
        # ax.plot(weights[:, 0], np.mean(biases, axis=1), losses, marker='o', linestyle='-', color='b')

        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Bias Value')
        ax.set_zlabel('Loss')
        ax.set_title('Adam Optimization Path')

        plt.show()
        plt.savefig('testing.png')

    def softmax(self, logits):
        z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return z / np.sum(z, axis=1, keepdims=True)

    def predict(self, data: pd.DataFrame):
        logits = np.dot(data, self.W.T) + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)
