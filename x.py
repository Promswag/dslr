import pandas as pd
from LogisticRegression import preprocessing
from LogisticRegression import standardize
from LogisticRegression import LogisticRegression


def main():
	df = pd.read_csv("datasets/dataset_train.csv")
	df = preprocessing(df, "Hogwarts House", keep_na=True)
	features = df.iloc[:-1, 2:-1]
	data = pd.DataFrame()

	for f in features:
		data.loc[:, f] = standardize(df[f])
	target = df["Hogwarts House"]

	print(data, '\n')

	lr = LogisticRegression(data, target)
	lr.gradient_descent()
	print(lr.W)
	print(lr.bias)

	pred = lr.predict(data)
	print('\n')
	print(pd.DataFrame(pred).value_counts())
	correct = pd.DataFrame(pred == target)
	print('\n')
	print(correct.value_counts())

	# l = LogisticRegression(data, target)
	# l.stochastic_gd()
	# print("weight:\n", l.W)
	# print(l.bias)

	# pred = l.predict(data)
	# print('\n')
	# print(pd.DataFrame(pred).value_counts())
	# correct = pd.DataFrame(pred == target)
	# print('\n')
	# print(correct.value_counts())


if __name__ == "__main__":
	main()
