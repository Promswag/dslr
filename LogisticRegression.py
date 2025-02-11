import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def fill_na(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
	numeric_features = df.select_dtypes(include='number').columns
	means = df.groupby(target_name)[numeric_features].mean()

	for c in df[target_name].unique():
		for f in numeric_features:
			df.loc[df[target_name] == c, f] = df.loc[df[target_name] == c, f].fillna(means.loc[c, f])

	return df

def outliers_clamping_by_std(df: pd.DataFrame, target_name: str, std_multiplier: float) -> pd.DataFrame:
	numeric_features = df.select_dtypes(include='number').columns
	means = df.groupby(target_name)[numeric_features].mean()
	stds = df.groupby(target_name)[numeric_features].std()

	def clamp(x, lower: float, upper: float):
		if x < lower:
			return lower
		if x > upper:
			return upper
		return x
	
	for c in df[target_name].unique():
		for f in numeric_features:
			lower = means.loc[c, f] - std_multiplier * stds.loc[c, f]
			upper = means.loc[c, f] + std_multiplier * stds.loc[c, f]
			df.loc[df[target_name] == c, f] = df.loc[df[target_name] == c, f].apply(lambda x: clamp(x, lower, upper))
	
	return df

def softmax(logits):
	z = np.exp(logits)
	return z / np.sum(z, axis=1, keepdims=True)

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
			self.losses = []
			self.bias_history = []
			self.weight_history = []
			self.bias = np.zeros(self.n_classes)
			self.W = np.zeros((self.n_classes, self.n_features))
		except Exception as e:
			print(f'{type(e).__name__}: {e}')
			return None
			  
	def reset(self):
		self.losses.clear()
		self.bias_history.clear()
		self.weight_history.clear()
		self.bias = np.zeros(self.n_classes)
		self.W = np.zeros((self.n_classes, self.n_features))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

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
	
	def compute_gradient(self, X, Y, m, save_cost):
		logits = np.dot(X, self.W.T) + self.bias
		pred = softmax(logits)
		if save_cost is True:
			self.losses.append(self.compute_loss(pred))
		np.add.at(pred, (np.arange(m), Y), -1)
		self.W -= self.learning_rate / m * np.dot(pred.T, X)
		self.bias -= self.learning_rate / m * np.sum(pred, axis=0)	
	
	def gradient_descent(self, save_cost: bool = False):
		for _ in range(self.epochs):
			self.compute_gradient(
				X = self.features,
				Y = self.target,
				m = self.m,
				save_cost = save_cost
				)

	def stochastic(self, save_cost: bool = False):
		for i in range(self.m):
			self.compute_gradient(
				X = self.features.iloc[[i]],
				Y = self.target.iloc[[i]],
				m = 1,
				save_cost = save_cost
				)

	def mini_batch(self, batch_size: int = 32, save_cost: bool = False):
		if batch_size <= 0:
			raise ValueError(f"Wrong value for batch size : {batch_size}")
		for _ in range(int(self.epochs / batch_size)):
			indices = np.random.permutation(self.m)
			for i in range(0, self.m, batch_size):
				batch_indices = indices[i:i + batch_size]
				self.compute_gradient(
					X = self.features.iloc[batch_indices],
					Y = self.target.iloc[batch_indices],
					m = len(batch_indices),
					save_cost = save_cost
					)


	def adam(self, beta1: float = 0.9, beta2: float = 0.999, batch_size: int = 32):
		epsilon = 1e-8
		m_w, v_w = np.zeros_like(self.W), np.zeros_like(self.W)
		m_b, v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
		t = 0

		for _ in range(self.epochs / batch_size):
			indices = np.random.permutation(self.m)
			for i in range(0, self.m, batch_size):
				batch_indices = indices[i:i + batch_size]
				X_batch, y_batch = self.features.iloc[batch_indices], self.target.iloc[batch_indices]
				logits = np.dot(X_batch, self.W.T) + self.bias
				probs = softmax(logits)
				m = len(y_batch)
				np.add.at(probs, (np.arange(m), y_batch), -1)
				grad_w = np.dot(probs.T, X_batch) / m
				grad_b = np.sum(probs, axis=0) / m

				t += 1
				m_w = beta1 * m_w + (1 - beta1) * grad_w
				v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
				m_b = beta1 * m_b + (1 - beta1) * grad_b
				v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

				m_w_hat = m_w / (1 - beta1 ** t)
				v_w_hat = v_w / (1 - beta2 ** t)
				m_b_hat = m_b / (1 - beta1 ** t)
				v_b_hat = v_b / (1 - beta2 ** t)

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
		plt.savefig('graphs/testing.png')
	
	def save_weights(self):
		try:
			path = 'datasets/weights.csv'
			weights = pd.DataFrame()
			weights[self.target.name] = pd.Series(self.classes.values())
			weights['Bias'] = pd.Series(self.bias)
			for i, f in enumerate(self.features):
				weights[f] = pd.Series(self.W[:, i])
			weights.to_csv(path, index=False)
			print(f"Weights have been saved in {path}")
		except Exception as e:
			print(f'{type(e).__name__}: {e}')

	def load_weights(self):
		try:
			weights = pd.read_csv('datasets/weights.csv')
			self.bias = weights['Bias'].values
			self.W = weights.iloc[:, 2:].values
		except Exception as e:
			print(f"{type(e).__name__} : {e}")

	def predict(self, data: pd.DataFrame, to_file: str = 'datasets/predictions.csv') -> pd.Series:
		logits = np.dot(data, self.W.T) + self.bias
		probabilies = softmax(logits)
		predictions = np.argmax(probabilies, axis=1)
		formatted = pd.Series(predictions, index=data.index, name=self.target.name).map(self.classes)

		if isinstance(to_file, str) is True:
			formatted.to_csv(to_file, index=True, index_label='Index')
			print(f"Predictions saved in {to_file}")

		return formatted
	
	@staticmethod
	def predict_from_weights(data: pd.DataFrame, weights: pd.DataFrame, to_file: str = 'datasets/predictions.csv') -> pd.Series:
		try:
			classes = {i: c for i, c in enumerate(weights.iloc[:, 0].values)}
			bias = weights.iloc[:, 1].values
			W = weights.iloc[:, 2:].values

			logits = np.dot(data, W.T) + bias
			prob = softmax(logits)
			pred = np.argmax(prob, axis=1)
			formatted = pd.Series(pred, index=data.index, name=weights.columns[0]).map(classes)

			if isinstance(to_file, str):
				formatted.to_csv(to_file, index=True, index_label='Index')
				print(f"Predictions saved in {to_file}")

			return formatted

		except Exception as e:
			print(f"{type(e).__name__} : {e}")
	

	def plot_sigmoid(self, df: pd.DataFrame):
		fig, axes = plt.subplots(ncols=self.n_features, figsize=(15, 10))

		for c in range(self.n_classes):
			for f in range(self.n_features):
				x_values = np.linspace(-5, 5, 100)
				
				logits_f = np.zeros((x_values.shape[0], self.n_classes))
				logits_f[:, c] = x_values * self.W[c, f] + self.bias[c]
				y_values = self.sigmoid(logits_f[:, c])
				
				ax = axes[f]
				ax.plot(x_values, y_values, label=f'{self.classes[c]}')
				ax.set_xlabel(df.columns[f])
				ax.set_ylabel('Probability')
				ax.grid(True)
		
		plt.tight_layout()
		plt.savefig('graphs/sigmoid')

	def realtime(self):
		self.reset()
		fig, axes = plt.subplots(ncols=3, figsize=(15,5))
		lines = []
		for f in range(self.n_features):
			ax = axes[f]
			ax.set_xlabel(self.features.columns[f])
			ax.set_ylabel('Probability')
			ax.set_xlim(-5, 5)
			ax.set_ylim(0, 1)
			c_lines = []
			for c in range(self.n_classes):
				line, = ax.plot([], [])
				c_lines.append(line)
			lines.append(c_lines)

		x = np.linspace(-10, 10, 100)

		def animate(i):
			if i >= self.epochs:
				return []
			self.compute_gradient(
				X = self.features,
				Y = self.target, 
				m = self.m,
				save_cost=True
				)
			
			for f in range(self.n_features):
				for c in range(self.n_classes):
					y = self.sigmoid(x * self.W[c, f] + self.bias[c])
					lines[f][c].set_data(x, y)

			return [l for line in lines for l in line]

		anim = animation.FuncAnimation(fig, animate, frames=self.epochs+1, interval=1, blit=True)
		anim.save('graphs/sigmoi_anim30.gif', writer='pillow', fps=30)
		plt.show()
