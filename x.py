from LogisticRegression import preprocessing
from LogisticRegression import outliers_clamping_by_std
from LogisticRegression import LogisticRegression
from StandardScaler import StandardScaler
import pandas as pd


def main():
	df = pd.read_csv("dataset_train.csv")
	df, classes = preprocessing(df, "Hogwarts House", keep_na=True)
	target = df["Hogwarts House"]
	features = [
		'Defense Against the Dark Arts',
		'Ancient Runes',
		'Charms',
	]

	scaler = StandardScaler()
	data = scaler.fit_transform(df[features])

	lr = LogisticRegression(data, target)
	lr.gradient_descent()
	# lr.mini_batch_gd()
	# lr.stochastic_gd()
	# print(lr.W)
	# print(lr.bias)

	pred = lr.predict(data[features])
	print(classes)
	print(pd.DataFrame(pred, columns=['House']).value_counts())
	correct = pd.DataFrame(pred == target)
	print(correct.value_counts())


	# xd = []
	# for i, p in enumerate(pred):
	# 	if target.iloc[i] != pred[i]:
	# 		xd.append([int(target.iloc[i]), int(pred[i])])
	# print(pd.DataFrame(xd, columns=['House', 'Guess']).value_counts().sort_values())

if __name__ == "__main__":
	main()
