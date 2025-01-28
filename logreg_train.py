import pandas as pd
import sys
from LogisticRegression import LogisticRegression as LR

OPTIONS = ['-help', '-model']
MODEL = ['batch', 'mini-batch', 'stochastic']

def main():
	help = False
	try:
		if len(sys.argv) < 2:
			raise Exception(f"Usage: {sys.argv[0]} [options] <train_dataset.csv>")
		
		if sys.argv[len(sys.argv) - 1][-4:] != ".csv":
			raise Exception("Please provide a training dataset with .csv format.")
		
		if len (sys.argv) > 2:

			skip = False
			for i in range(1, len(sys.argv) - 1):
				if skip is True:
					skip = False
					continue

				if sys.argv[i][0] != '-':
					raise Exception("Options should start with an hyphen.")
				
				if not OPTIONS.__contains__(sys.argv[i].lower()):
					raise Exception(f"Invalid option : {sys.argv[i]}")
				
				if sys.argv[i].lower() == '-help':
					help = True

				if sys.argv[i].lower() == '-model':
					if not MODEL.__contains__(sys.argv[i + 1].lower()):
						raise Exception(f"Invalid parameter for -model : {sys.argv[i + 1]}")
					skip = True

		if help is True:
			print(f"Types for the gradient descent: {' '.join(MODEL)}")
				
		features = [
			'Defense Against the Dark Arts',
			'Ancient Runes',
			'Charms',
		]
		target = 'Hogwarts House'
		# df = pd.read_csv(sys.argv[1])
		# lr = LR(df[features], df[target])
		
	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

