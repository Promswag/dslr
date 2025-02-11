import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from LogisticRegression import LogisticRegression
from StandardScaler import StandardScaler

def main():
	try:
		DATASET = load_breast_cancer(as_frame=True)
		# DATASET = load_iris(as_frame=True)
		target = 'target'
		DATASET.frame[target] = DATASET.target_names[DATASET.target]
		df = pd.DataFrame(DATASET.frame)

		VALIDATION = df.sample(frac=0.33)
		TRAIN = df.drop(VALIDATION.index)

		features = df.columns.difference([target])

		scaler = StandardScaler()
		TRAIN[features] = scaler.fit_transform(TRAIN[features])
		VALIDATION[features] = scaler.transform(VALIDATION[features])

		lr = LogisticRegression(TRAIN[features], TRAIN[target])
		lr.gradient_descent()
		pred = lr.predict(VALIDATION[features])
		print((pred == VALIDATION[target]).value_counts())
		print(accuracy_score(pred, VALIDATION[target]))
	except Exception as e:
		print(f"{type(e).__name__} : {e}")

if __name__ == "__main__":
	main()

