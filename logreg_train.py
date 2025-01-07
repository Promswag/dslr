import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def standardize(lst):
	mean = sum(lst) / len(lst)
	std = (sum((abs(v - mean) ** 2) for v in lst) / len(lst)) ** 0.5
	return [(v - mean) / std for v in lst]

def softmax(arr):
	z = np.exp(arr - np.max(arr, axis=1, keepdims=True))
	return z / np.sum(z, axis=1, keepdims=True)

def main():
	# try:
		data = pd.read_csv("dataset_train.csv")
		features = [
			# 'Arithmancy',
			# 'Astronomy',
			# 'Herbology',
			'Defense Against the Dark Arts',
			# 'Divination',
			# 'Muggle Studies',
			'Ancient Runes',
			# 'History of Magic',
			# 'Transfiguration',
			# 'Potions',
			# 'Care of Magical Creatures',
			'Charms',
			# 'Flying',
		]
		X = data[features + ["Hogwarts House"]]
		print(X)
		means = X.groupby("Hogwarts House").mean()
		X = X.apply(lambda x: x.fillna(means.loc[x["Hogwarts House"]]), axis=1)
		print(means)
		# X.fillna(X.mean(), inplace=True)
		# data.fillna(data.mean(), inplace=True)
		# data.dropna(inplace=True)
		num_map = {
			'Gryffindor': 0,
			'Slytherin': 1,
			'Hufflepuff': 2,
			'Ravenclaw': 3
		}
		data['Hogwarts House'] = data['Hogwarts House'].map(num_map)
		print(data['Hogwarts House'].value_counts())
		print(f'{len(data)}\n')
		for f in features:
			X[f] = standardize(X[f].values)

		X = X[features]
		Y = data['Hogwarts House']
		L = 0.01

		m, n_features = X.shape
		n_classes = 4
		bias = np.zeros(n_classes)
		W = np.zeros((n_classes, n_features))

		for _ in range(1000):
			logits = np.dot(X, W.T) + bias

			pred = softmax(logits)
			grad = pred
			grad[range(m), Y] -=1

			grad_W = np.dot(grad.T, X) / m
			W -= L * grad_W

			grad_b = np.sum(grad, axis=0) / m
			bias -= L * grad_b

		print(W)
		print(bias)

		logits = np.dot(X, W.T) + bias
		prob = softmax(logits)
		pred = np.argmax(prob, axis=1)

		print('\n')
		print(pd.DataFrame(pred).value_counts())
		print('\n')
		print(prob[0])
		print(pred[0])

		correct = pd.DataFrame(pred == Y)
		print('\n')
		print(correct.value_counts())

		# xd = []
		# for i, p in enumerate(pred):
		# 	if Y.iloc[i] != pred[i]:
		# 		xd.append([int(Y.iloc[i]), int(pred[i])])
		# print(xd)
		# print(pd.DataFrame(xd).value_counts().sort_values())

	# except Exception as e:
	# 	print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

