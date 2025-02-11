import pandas as pd
import sys
from sklearn.metrics import accuracy_score

def main():
	# df = pd.read_csv("datasets/dataset_train.csv")
	# df['Hogwarts House'].to_csv("datasets/truth.csv", index_label='Index')
	try:
		predictions = pd.read_csv(sys.argv[1], index_col='Index')
		truth = pd.read_csv(sys.argv[2], index_col='Index')
		print(accuracy_score(predictions, truth))
	except Exception as e:
		print(f"{type(e).__name__} : {e}")

if __name__ == "__main__":
	main()

