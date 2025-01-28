import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def standardize(lst):
	mean = sum(lst) / len(lst)
	std = (sum((abs(v - mean) ** 2) for v in lst) / len(lst)) ** 0.5
	return [(v - mean) / std for v in lst]

def main():
	# try:
		data = pd.read_csv("datasets/dataset_train.csv")
		data.dropna(inplace=True)
		features = [
			'Arithmancy',
			'Astronomy',
			'Herbology',
			'Defense Against the Dark Arts',
			'Divination',
			'Muggle Studies',
			'Ancient Runes',
			'History of Magic',
			'Transfiguration',
			'Potions',
			'Care of Magical Creatures',
			'Charms',
			'Flying',
		]
		num_map = {
			'Gryffindor': 0,
			'Slytherin': 1,
			'Hufflepuff': 2,
			'Ravenclaw': 3
		}
		data['Hogwarts House'] = data['Hogwarts House'].map(num_map)
		print(data['Hogwarts House'].value_counts())
		print(len(data))
		for f in features:
			data[f] = standardize(data[f].values)
		# print(data)

		Y = data['Hogwarts House']
		X = data[['Divination']]
		L = 0.01
		t0 = 0
		tn = {f: 0 for f in X.columns}
		m = len(Y)
		epsilon = 1e-15
		for _ in range(1000):
			w = t0 + sum(tn[f] * X[f].values for f in tn.keys())
			pred = 1 / (1 + np.exp(np.clip(-w, -709, 709)))
			# cost = - (1 / m) * sum(Y * np.log(pred + epsilon) + (1 - Y) * np.log(1 - pred + epsilon))
			t0 -= L * (1 / m) * sum(pred - Y)
			for f in X.columns:
				tn[f] -= L * (1 / m) * sum((pred - Y) * X[f].values)
	
		print(sum(1 for p in pred if p >= 0.5))
		for i, p in enumerate(pred):
			if p > 0.5:
				pred[i] = 1
			else:
				pred[i] = 0
		
		print((data['Hogwarts House'] == pred).value_counts())
			# for x in pred:
		# 	if x < 0.9:
		# 		print(x)
		# print(pred)
		# print(pred)



	# except Exception as e:
	# 	print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

