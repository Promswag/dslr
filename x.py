import pandas as pd
from LogisticRegression import preprocessing
from LogisticRegression import standardize
from LogisticRegression import LogisticRegression


def main():
	df = pd.read_csv("dataset_train.csv")
	df = preprocessing(df, "Hogwarts House", keep_na=False)
	features = [
		'Defense Against the Dark Arts',
		'Ancient Runes',
		'Charms',
	]
	data = df[features]
	lr = LogisticRegression(data, df["Hogwarts House"])
	lr.gradient_descent()
	print(lr.bias)

	pred = lr.predict(data)
	print(pred)


if __name__ == "__main__":
	main()
