# import matplotlib.pyplot as plt
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
			self.costs = []
			self.bias = np.zeros(self.n_classes)
			self.W = np.zeros((self.n_classes, self.n_features))
		except Exception as e:
			print(f'{type(e).__name__}: {e}')
			return None
		
	def gradient_descent(self):
		for i in range(self.epochs):
			logits = np.dot(self.features, self.W.T) + self.bias
			pred = self.softmax(logits)
			pred[range(self.m), self.target] -= 1
			self.W -= self.learning_rate * (np.dot(pred.T, self.features) / self.m)
			self.bias -= self.learning_rate * (np.sum(pred, axis=0) / self.m)


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