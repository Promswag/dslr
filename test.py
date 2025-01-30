from LogisticRegression import fill_na
from LogisticRegression import outliers_clamping_by_std
from LogisticRegression import LogisticRegression
from StandardScaler import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import time


def main():
	df = pd.read_csv("dataset_train.csv")
	df.set_index('Index', inplace=True)
	df = fill_na(df, "Hogwarts House")
	features = [
		'Defense Against the Dark Arts',
		'Ancient Runes',
		'Charms',
	]
	target = 'Hogwarts House'

	VALIDATION = df.sample(frac=0.25)
	VALIDATION.to_csv('validation.csv')

	df.drop(VALIDATION.index, inplace=True)
	df = outliers_clamping_by_std(df, target, 2)

	scaler = StandardScaler()
	df[features] = scaler.fit_transform(df[features])
	VALIDATION[features] = scaler.transform(VALIDATION[features])

	lr = LogisticRegression(df[features], df[target])
	lr.gradient_descent()
	lr.save_weights()
	weights = pd.read_csv('weights.csv')
	pred = LogisticRegression.predict_from_weights(VALIDATION[features], weights)
	correct = (pred == VALIDATION[target]).value_counts()
	print(correct.iloc[0] / len(pred))

	lr.reset()
	lr.stochastic()
	lr.save_weights()
	weights = pd.read_csv('weights.csv')
	pred = LogisticRegression.predict_from_weights(VALIDATION[features], weights)
	correct = (pred == VALIDATION[target]).value_counts()
	print(correct.iloc[0] / len(pred))

	lr.reset()
	lr.mini_batch()
	lr.save_weights()
	weights = pd.read_csv('weights.csv')
	pred = LogisticRegression.predict_from_weights(VALIDATION[features], weights)
	correct = (pred == VALIDATION[target]).value_counts()

			
	print(correct.iloc[0] / len(pred))

if __name__ == "__main__":
	main()
