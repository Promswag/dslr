import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import numpy as np
import pandas as pd


def fill_na(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
	numeric_features = df.select_dtypes(include='number').columns
	means = df.groupby(target_name)[numeric_features].mean()

	for c in df[target_name].unique():
		for f in numeric_features:
			df.loc[df[target_name] == c, f] = df.loc[df[target_name] == c, f].fillna(means.loc[c, f])

	return df

# def preprocessing(df: pd.DataFrame, target_name: str, keep_na: bool=False) -> pd.DataFrame:
# 	data = df.select_dtypes(include='number')
# 	classes = {c: i for i, c in enumerate(df[target_name].unique())}
# 	data[target_name] = df[target_name].map(classes)

# 	if keep_na is False:
# 		data.dropna(inplace=True)
# 	else:
# 		means = data.groupby(target_name).mean()
# 		data = data.apply(lambda x: x.fillna(means.loc[x[target_name]]), axis=1)
# 		data[target_name] = data[target_name].astype(int)

# 	return data, classes

def outliers_clamping_by_std(df: pd.DataFrame, target_name: str, std_multiplier: float) -> pd.DataFrame:
	means = df.groupby(target_name).mean()
	stds = df.groupby(target_name).std()

	def clamp(x, lower: float, upper: float):
		if x < lower:
			return lower
		if x > upper:
			return upper
		return x
	
	for c in df[target_name].unique():
		for f in df.columns.difference([target_name]):
			lower = means.loc[c, f] - std_multiplier * stds.loc[c, f]
			upper = means.loc[c, f] + std_multiplier * stds.loc[c, f]
			df.loc[df[target_name] == c, f] = df.loc[df[target_name] == c, f].apply(lambda x: clamp(x, lower, upper))
	
	return df


class LogisticRegression():
	def __init__(self, features: pd.DataFrame, target: pd.Series, learning_rate: float = 0.01, epochs: int = 1000):
		try:
			if (features.shape[0] != target.shape[0]):
				raise ValueError(
					"LogisticRegression: Features and Target must be of the same shape.")
			self.m, self.n_features = features.shape
			self.n_classes = len(target.unique())
			self.classes = {i: c for i, c in enumerate(target.unique())}
			self.target = target.map({v: k for k, v in self.classes.items()})
			self.features = features
			self.learning_rate = learning_rate
			self.epochs = epochs
			self.costs = []
			self.bias = np.zeros(self.n_classes)
			self.W = np.zeros((self.n_classes, self.n_features))
		except Exception as e:
			print(f'{type(e).__name__}: {e}')
			return None
	
	def reset(self):
		self.costs = []
		self.bias = np.zeros(self.n_classes)
		self.W = np.zeros((self.n_classes, self.n_features))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

	def plot_sigmoid(self, df: pd.DataFrame):
		""" Tracer la fonction sigmoïde pour chaque feature et chaque classe """
		fig, axes = plt.subplots(ncols=self.n_features, figsize=(15, 10))

		for c in range(self.n_classes):  # Boucle sur chaque classe
			for f in range(self.n_features):  # Boucle sur chaque feature
				# Générer des valeurs pour la feature f
				x_values = np.linspace(np.min(df.iloc[:, f]), np.max(df.iloc[:, f]), 1000)
				
				# Modifier les logits en fonction de la feature
				logits_f = np.zeros((x_values.shape[0], self.n_classes))
				
				# Pour chaque valeur de x_values, appliquer le poids de la feature f pour la classe c
				logits_f[:, c] = x_values * self.W[c, f] + self.bias[c]  # Appliquer le poids et le biais pour la classe c
				
				# Appliquer la fonction sigmoïde pour obtenir la probabilité
				y_values = self.sigmoid(logits_f[:, c])
				
				# Tracer la probabilité en fonction de la feature
				ax = axes[f]
				ax.plot(x_values, y_values, label=f'{self.classes[c]}')
				ax.set_xlabel(df.columns[f])
				ax.set_ylabel('Probabilité')
				ax.grid(True)
		
		plt.tight_layout()
		plt.show()

	def gradient_descent(self):
		for _ in range(self.epochs):
			logits = np.dot(self.features, self.W.T) + self.bias
			pred = self.softmax(logits)
			np.add.at(pred, (np.arange(self.m), self.target), -1)
			self.W -= self.learning_rate / self.m * np.dot(pred.T, self.features)
			self.bias -= self.learning_rate * (np.sum(pred, axis=0) / self.m)

	def stochastic(self):
		for i in range(self.m):
			logits = np.dot(self.features.iloc[[i]], self.W.T) + self.bias
			pred = self.softmax(logits)
			pred[[0], self.target.iloc[[i]]] -= 1
			self.W -= self.learning_rate * np.dot(pred.T, self.features.iloc[[i]])
			self.bias -= self.learning_rate * np.sum(pred, axis=0)

	def mini_batch(self):
		loop = 20
		batch_size = 50
		offset = self.m % batch_size
		batch_count = int(self.m / batch_size)

		for _ in range(loop):
			shuffled_features = self.features.sample(frac=1)
			shuffled_target = self.target.reindex(shuffled_features.index)
			for i in range(batch_count):
				batch_features = shuffled_features.iloc[i * batch_size: (i + 1) * batch_size + (offset if i == batch_count - 1 else 0)]
				batch_target = shuffled_target.iloc[i * batch_size: (i + 1) * batch_size + (offset if i == batch_count - 1 else 0)]

				logits = np.dot(batch_features, self.W.T) + self.bias
				pred = self.softmax(logits)
				pred[range(batch_size + (offset if i == batch_count - 1 else 0)), batch_target] -= 1
				self.W -= self.learning_rate * np.dot(pred.T, batch_features)
				self.bias -= self.learning_rate * np.sum(pred, axis=0)

	def softmax(self, logits):
		# z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
		z = np.exp(logits)
		return z / np.sum(z, axis=1, keepdims=True)
	
	def predict(self, data: pd.DataFrame, to_file: bool = False) -> pd.Series:
		logits = np.dot(data, self.W.T) + self.bias
		probabilies = self.softmax(logits)
		predictions = np.argmax(probabilies, axis=1)
		formatted = pd.Series(predictions, index=self.target.index, name=self.target.name).map(self.classes)

		if to_file is True:
			formatted.to_csv('houses.csv', index=True, index_label='Index')

		return formatted
	
	def predict_from_weights(self, data: pd.DataFrame, path: str = 'weights.csv', to_file: bool = False) -> pd.Series:
		self.load_weights(path)
		return self.predict(data, to_file)
	
	def load_weights(self, path: str = 'weights.csv'):
		try:
			weights = pd.read_csv(path)
			self.bias = weights['Bias'].values
			self.W = weights.iloc[:, 2:].values
		except Exception as e:
			print(f"{type(e).__name__} : {e}")
	
	def save_weights(self, path: str = 'weights.csv'):
		try:
			weights = pd.DataFrame()
			weights['Class'] = pd.Series(self.target.unique())
			weights['Bias'] = pd.Series(self.bias)
			for i, f in enumerate(self.features):
				weights[f] = pd.Series(self.W[:, i])
			weights.to_csv(path, index=False)
		except Exception as e:
			print(f'{type(e).__name__}: {e}')