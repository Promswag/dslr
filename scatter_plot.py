import matplotlib.pyplot as plt
import pandas as pd


def main():
	data = pd.read_csv("dataset_train.csv")
	data['Best Hand'] = (data['Best Hand'] == 'Right').astype(int)
	print(data)
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
	data.loc[data['Hogwarts House'] == 'Gryffindor', 'Hogwarts House'] = 'r'
	data.loc[data['Hogwarts House'] == 'Slytherin', 'Hogwarts House'] = 'g'
	data.loc[data['Hogwarts House'] == 'Hufflepuff', 'Hogwarts House'] = 'y'
	data.loc[data['Hogwarts House'] == 'Ravenclaw', 'Hogwarts House'] = 'b'
	print(data)

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
	pass

if __name__ == "__main__":
	main()

