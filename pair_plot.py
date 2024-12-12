import matplotlib.pyplot as plt
import pandas as pd
from textwrap import wrap


def main():
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
		]
	color_map = {
		'Gryffindor': 'r',
		'Slytherin': 'g',
		'Hufflepuff': 'y',
		'Ravenclaw': 'b',
		}
	data['Hogwarts House'] = data['Hogwarts House'].map(color_map)

	fig, axes = plt.subplots(nrows=len(features), ncols=len(features))
	fig.suptitle("Pair plot of Hogwarts Houses's features", fontsize=24)
	plt.get_current_fig_manager().full_screen_toggle()

	for j in range(len(features)):
		for i in range(len(features)):
			ax = axes[j, i]
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.tick_params(bottom=False, left=False)
			if j == 0:
				ax.set_xlabel('\n'.join(wrap(features[i], 16)), fontsize=8)
				ax.xaxis.set_label_position('top')
			if i == 0:
				ax.set_ylabel('\n'.join(wrap(features[j], 16)), rotation=0, fontsize=8, ha='right', va='center')
			if i == j:
				continue
			ax.scatter(data[features[j]], data[features[i]], c=data['Hogwarts House'])

	plt.subplots_adjust(left=0.075, bottom=0.01, right=0.99, top=0.90, wspace=0, hspace=0)
	plt.show()


if __name__ == "__main__":
	main()

