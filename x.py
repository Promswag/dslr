import pandas as pd
from LogisticRegression import preprocessing
from LogisticRegression import standardize
from LogisticRegression import LogisticRegression


def main():
	df = pd.read_csv("dataset_train.csv")
	df, classes = preprocessing(df, "Hogwarts House", keep_na=False)
	features = [
		# 'History of Magic',
		'Defense Against the Dark Arts',
		'Ancient Runes',
		# 'Divination',
		'Charms',
	]
	data = pd.DataFrame()
	for f in features:
		data.loc[:, f] = standardize(df[f])
	target = df["Hogwarts House"]

	# print(data)

	lr = LogisticRegression(data, target)
	lr.gradient_descent()
	lr.plot_sigmoid(data[features], classes)
	print(lr.W)
	print(lr.bias)

	pred = lr.predict(data)
	print('\n')
	print(pd.DataFrame(pred).value_counts())
	correct = pd.DataFrame(pred == target)
	print('\n')
	print(correct.value_counts())

	# l = LogisticRegression(data, target)
	# l.mini_batch_gd()
	# # l.stochastic_gd()
	# print(l.W)
	# print(l.bias)

	# pred = l.predict(data)
	# print('\n')
	# print(pd.DataFrame(pred).value_counts())
	# correct = pd.DataFrame(pred == target)
	# print('\n')
	# print(correct.value_counts())

	# xd = []
	# for i, p in enumerate(pred):
	# 	if target.iloc[i] != pred[i]:
	# 		xd.append([int(target.iloc[i]), int(pred[i])])
	# print(pd.DataFrame(xd, columns=['House', 'Guess']).value_counts().sort_values())
	# print(classes)

if __name__ == "__main__":
	main()
