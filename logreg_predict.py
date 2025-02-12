import pandas as pd
import sys
from LogisticRegression import LogisticRegression
from StandardScaler import StandardScaler

def main():
	try:
		if len(sys.argv) < 2 or len(sys.argv) > 3: 
			raise Exception(f"Usage: {sys.argv[0]} <dataset.csv>")
		
		for file in sys.argv[1:]:
			if file[-4:] != ".csv":
				raise Exception(f"Wrong file format for {file}, expected: .csv")
		
		weights = pd.read_csv('datasets/weights.csv')
		features = [f for f in weights.columns[2:]]

		df = pd.read_csv(sys.argv[-1], index_col='Index')[features]

		scaler = StandardScaler().from_file()
		df[features] = scaler.transform(df[features])
		df = df.fillna(0)
		
		pred = LogisticRegression.predict_from_weights(df[features], weights, to_file='datasets/houses.csv')
		
	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

