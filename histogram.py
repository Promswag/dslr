import matplotlib.pyplot as plt
import pandas as pd


def main():
	data = pd.read_csv("dataset_train.csv")
	# data['Best Hand'] = (data['Best Hand'] == 'Right').astype(int)
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
	gryffindor = data[data['Hogwarts House'] == 'Gryffindor']
	slytherin = data[data['Hogwarts House'] == 'Slytherin']
	hufflepuff = data[data['Hogwarts House'] == 'Hufflepuff']
	ravenclaw = data[data['Hogwarts House'] == 'Ravenclaw']

	fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(16,8))
	fig.suptitle("Histogram of the Hogwarts House's courses", fontsize=24)

	for i, feature in enumerate(features):
		ax = axes[int(i / 5), i % 5]
		ax.set_title(feature)
		ax.hist(gryffindor[feature], bins=30, color='r', alpha=0.4)
		ax.hist(slytherin[feature], bins=30, color='g', alpha=0.4)
		ax.hist(hufflepuff[feature], bins=30, color='y', alpha=0.4)
		ax.hist(ravenclaw[feature], bins=30, color='b', alpha=0.4)
		ax.set_xticklabels('')
		ax.set_yticklabels('')

	axes[2, 3].remove()
	axes[2, 4].remove()

	fig.legend(labels=['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw'],
			loc='center', bbox_to_anchor=(0.68, 0.165), fontsize=16)
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85)
	plt.show()
	pass

if __name__ == "__main__":
	main()

