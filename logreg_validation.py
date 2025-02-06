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
				
		features = [
			'Defense Against the Dark Arts',
			'Ancient Runes',
			'Charms',
		]
		target = 'Hogwarts House'

		df = pd.read_csv(sys.argv[-1], index_col='Index')[features + [target]]
		print(df)
		scaler = StandardScaler().from_file()
		df[features] = scaler.transform(df[features])
		pred = LogisticRegression.predictXD(df[features])

		correct = (pred == df[target]).value_counts()
		print(correct)
		print(correct.iloc[0] / len(pred))

		
	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

