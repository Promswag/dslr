import matplotlib.pyplot as plt
import pandas as pd


def main():
	try:
		data = pd.read_csv("datasets/dataset_train.csv")
		color_map = {
			'Gryffindor': 'r',
			'Slytherin': 'g',
			'Hufflepuff': 'y',
			'Ravenclaw': 'b',
			}
		data['Hogwarts House'] = data['Hogwarts House'].map(color_map)

		fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,8))
		fig.suptitle("Similar features of the Hogwarts House", fontsize=24)

		ax1.scatter(data['Astronomy'], data['Defense Against the Dark Arts'], c=data['Hogwarts House'], alpha=0.4)
		ax1.set_xlabel('Astronomy')
		ax1.set_ylabel('Defense Against the Dark Arts')

		ax2.scatter(data['History of Magic'], data['Transfiguration'], c=data['Hogwarts House'], alpha=0.4)
		ax2.set_xlabel('History of Magic')
		ax2.set_ylabel('Transfiguration')

		plt.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.85, wspace=0.25)
		plt.show()

	except Exception as e:
		print(f'{type(e).__name__} : {e}')

if __name__ == "__main__":
	main()

