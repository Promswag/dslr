import matplotlib.pyplot as plt
import pandas as pd
from textwrap import wrap
import seaborn as sns # type: ignore


def main():
	try:
		data = pd.read_csv("dataset_train.csv")
		features = [
			'Arithmancy',
			'Astronomy',
			'Herbology',
			'Defense Against the Dark Arts',
			'Divination',
			'Muggle Studies',
			'Ancient Runes',
			'History of Magic',
			'Transfiguration',
			'Potions',
			'Care of Magical Creatures',
			'Charms',
			'Flying',
			'Hogwarts House',
			]
		color_map = {
			'Gryffindor': 'r',
			'Slytherin': 'g',
			'Hufflepuff': 'y',
			'Ravenclaw': 'b',
			}

		pair_plot = sns.pairplot(data[features], hue="Hogwarts House", palette=color_map)
		pair_plot.figure.suptitle("Pair plot of Hogwarts Houses's features", fontsize=24)
		pair_plot.legend.set_bbox_to_anchor([0.075,0.925])
		for ax in pair_plot.axes.flatten():
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_ylabel('\n'.join(wrap(ax.get_ylabel(), 16)), rotation=0, fontsize=8, ha='right', va='center')
			ax.set_xlabel('\n'.join(wrap(ax.get_xlabel(), 16)), fontsize=8)
		plt.subplots_adjust(left=0.075, bottom=0.05, right=0.99, top=0.85, wspace=0, hspace=0)
		plt.show()

	except Exception as e:
		print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
	main()

