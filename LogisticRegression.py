import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.animation as animation
import numpy as np
import pandas as pd


def preprocessing(df: pd.DataFrame, target_name: str, keep_na: bool=False) -> pd.DataFrame:
    data = df.select_dtypes(include='number')
    classes = {c: i for i, c in enumerate(df[target_name].unique())}
    data[target_name] = df[target_name].map(classes)

    if keep_na is False:
        data.dropna(inplace=True)
    else:
        means = data.groupby(target_name).mean()
        data = data.apply(lambda x: x.fillna(means.loc[x[target_name]]), axis=1)
        data[target_name] = data[target_name].astype(int)

    return data


def standardize(lst: pd.Series) -> pd.Series:
    mean = sum(lst) / len(lst)
    std = (sum((abs(value - mean) ** 2) for value in lst) / (len(lst) - 1)) ** 0.5
    return pd.Series(((lst - mean) / std), index=lst.index)


class LogisticRegression():
    def __init__(self, features: pd.DataFrame, target: pd.Series, learning_rate: float = 0.01, epochs: int = 1000):
        try:
            if (features.shape[0] != target.shape[0]):
                raise ValueError(
                    "LogisticRegression: Features and Target must be of the same shape.")
            self.m, self.n_features = features.shape
            self.n_classes = len(target.unique())
            self.features = features
            self.target = target
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.losses = []
            self.bias = np.random.random(self.n_classes)
            self.W = np.random.random((self.n_classes, self.n_features))
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            return None

    def gradient_descent(self):
        for _ in range(self.epochs):
            logits = np.dot(self.features, self.W.T) + self.bias
            pred = self.softmax(logits)
            loss_pred = pred.copy()
            pred[range(self.m), self.target] -= 1
            self.W -= self.learning_rate * (np.dot(pred.T, self.features) / self.m)
            self.bias -= self.learning_rate * (np.sum(pred, axis=0) / self.m)
            self.losses.append(self.compute_loss(loss_pred))

        plt.plot(range(len(self.losses)), self.losses, label="Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Cross-Entropy)")
        plt.title("Error Curve During Training")
        plt.legend()
        plt.savefig('1')

        sample_input = self.features.iloc[0:-1]  # Take the first sample
        probabilities = self.softmax(np.dot(sample_input, self.W.T) + self.bias)
        plt.bar(range(len(probabilities[0])), probabilities[0])
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Class Probability Distribution for a Sample Input")
        plt.savefig('2')

        # x_min, x_max = self.features.iloc[:, 0].min() - 1, self.features.iloc[:, 0].max() + 1
        # y_min, y_max = self.features.iloc[:, 1].min() - 1, self.features.iloc[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
        #                     np.linspace(y_min, y_max, 100))
        # Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        # Z = Z.reshape(xx.shape)
        # plt.contourf(xx, yy, Z, alpha=0.3)
        # plt.scatter(self.features.iloc[:, 0], self.features.iloc[:, 1], c=self.target, edgecolor='k')
        # plt.xlabel("Feature 1")
        # plt.ylabel("Feature 2")
        # plt.title("Decision Boundary")
        # plt.savefig('3')

        y_true = self.target
        y_pred = self.predict(self.features)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig('4')

    x = np.linspace(-10, 10, 100)
    logits = np.array([x, -x, x/2, -x/2]).T
    softmax_values = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    plt.plot(x, softmax_values[:, 0], label="Class 1")
    plt.plot(x, softmax_values[:, 1], label="Class 2")
    plt.plot(x, softmax_values[:, 2], label="Class 3")
    plt.plot(x, softmax_values[:, 3], label="Class 4")
    plt.xlabel("Feature Value")
    plt.ylabel("Probability")
    plt.title("Softmax Output for 4 Classes")
    plt.legend()
    plt.savefig('5')

    def compute_loss(self, pred):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.log(pred[np.arange(self.m), self.target]))
        return loss

    def stochastic_gd(self):
        for i in range(self.m):
            logits = np.dot(self.features.iloc[[i]], self.W.T) + self.bias
            pred = self.softmax(logits)
            pred[[0], self.target.iloc[[i]]] -= 1
            self.W -= self.learning_rate * np.dot(pred.T, self.features.iloc[[i]])
            self.bias -= self.learning_rate * np.sum(pred, axis=0)

    def mini_batch_gd():
        pass

    def softmax(self, logits):
        z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return z / np.sum(z, axis=1, keepdims=True)

    def predict(self, data: pd.DataFrame):
        logits = np.dot(data, self.W.T) + self.bias
        probabilies = self.softmax(logits)
        predictions = np.argmax(probabilies, axis=1)
        return predictions
