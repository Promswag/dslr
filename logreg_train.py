import pandas as pd
import sys
from LogisticRegression import LogisticRegression
from LogisticRegression import fill_na
from StandardScaler import StandardScaler

OPTIONS = ['-help', '-model']
MODEL = {
	'batch': LogisticRegression.gradient_descent,
	'mini-batch': LogisticRegression.mini_batch,
	'stochastic': LogisticRegression.stochastic,
	'adam': LogisticRegression.adam
}

def main():
	help = False
	model = 'batch'
	try:
		if len(sys.argv) < 2:
			raise Exception(f"Usage: {sys.argv[0]} [options] <train_dataset.csv>")
		
		if sys.argv[-1][-4:] != ".csv":
			raise Exception("Please provide a training dataset with .csv format.")
		
		if len (sys.argv) > 2:
			skip = False
			for i in range(1, len(sys.argv) - 1):
				if skip is True:
					skip = False
					continue

				if sys.argv[i][0] != '-':
					raise Exception("Options should start with an hyphen.")
				
				if not OPTIONS.__contains__(sys.argv[i]):
					raise Exception(f"Invalid option : {sys.argv[i]}")
				
				if sys.argv[i] == '-help':
					help = True

				if sys.argv[i] == '-model':
					if not MODEL.__contains__(sys.argv[i + 1]):
						raise Exception(f"Invalid parameter for -model : {sys.argv[i + 1]}")
					model = sys.argv[i + 1]
					skip = True

		if help is True:
			print(f"Models for the gradient descent: {' '.join(MODEL)}")
				
		features = [
			'Defense Against the Dark Arts',
			'Ancient Runes',
			'Charms',
		]
		target = 'Hogwarts House'

		df = pd.read_csv(sys.argv[-1])[features + [target]]

		df = fill_na(df, target)
		scaler = StandardScaler()
		df[features] = scaler.fit_transform(df[features])
		scaler.save_to_file()

		lr = LogisticRegression(df[features], df[target])
		MODEL[model](lr)
		lr.save_weights()
	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

