import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

class DataPreprocessing():
	pass

class LogisticRegression():
	# def __init__(self, X: np.ndarray[float, int], Y: np.ndarray[float, int], learning_rate: float = 0.1, epochs: int = 1000):
	def __init__(self, data: pd.DataFrame, target: str, learning_rate: float = 0.1, epochs: int = 1000):

		try:
			if len(X) != len(Y):
				raise ValueError(
					"LogisticRegression: X and Y must be equal length.")
			if len(X) < 3:
				raise ValueError(
					"LogisticRegression: X and Y must be of length of 3 or more.")
			for x in X:
				if not isinstance(x, (float, int)):
					raise ValueError(
						"LogisticRegression: X must contain float or int only.")
			for y in Y:
				if not isinstance(y, (float, int)):
					raise ValueError(
						"LogisticRegression: Y must contain float or int only.")
			self.m = float(len(X))
			self.X = X
			self.Y = Y
			self.X_n = np.array([self.standardize(x, X) for x in X])
			self.Y_n = np.array([self.standardize(y, Y) for y in Y])
			self.learning_rate = learning_rate
			self.epochs = epochs
			self.costs = []
		except Exception as e:
			print(f'{type(e).__name__}: {e}')
			return None
		
	def standardize(item, lst):
		item - lst.mean() / lst.std()
		mean = sum(lst) / len(lst)
		std = (sum((abs(v - mean) ** 2) for v in lst) / len(lst)) ** 0.5
		return [(v - mean) / std for v in lst]
		
