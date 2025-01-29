from LogisticRegression import fill_na
# from LogisticRegression import outliers_clamping_by_std
from LogisticRegression import LogisticRegression
from StandardScaler import StandardScaler
import pandas as pd
import time


def main():
	df = pd.read_csv("dataset_train.csv")
	# df.dropna(inplace=True)
	test_df = pd.read_csv("dataset_test.csv")
	df = fill_na(df, "Hogwarts House")
	target = df["Hogwarts House"]
	features = [
		'Defense Against the Dark Arts',
		'Ancient Runes',
		'Charms',
	]

	# kekeke = {c: i for i, c in enumerate(target.unique())}
	# print(kekeke)
	# features = [
	# 	'Arithmancy',
	# 	'Astronomy',
	# 	'Herbology',
	# 	'Defense Against the Dark Arts',
	# 	'Divination',
	# 	'Muggle Studies',
	# 	'Ancient Runes',
	# 	'History of Magic',
	# 	'Transfiguration',
	# 	'Potions',
	# 	'Care of Magical Creatures',
	# 	'Charms',
	# 	'Flying',
	# ]
	# return
	scaler = StandardScaler()
	data = scaler.fit_transform(df[features])
	# print(scaler.features)
	# print(scaler.mean)
	# print(scaler.std)

	# scaler.from_file('scaler.csv')
	# print(scaler.features)
	# print(scaler.mean)
	# print(scaler.std)
	# return
	# scaler.save_to_file()
	# scaled_test_df = scaler.transform(test_df[features])
	# scaler2 = StandardScaler()
	# scaled_test_df = scaler2.fit_transform(test_df[features])
	# print(scaled_test_df)
	# print(test_df[features])
	# print(scaled_test_df)

	lr = LogisticRegression(data, target)
	# now = time.time_ns()
	lr.gradient_descent()
	# then = time.time_ns()
	# print(f'{(then - now)/1e9:.2f}')
	# lr.plot_sigmoid(scaled_test_df[features], classes)
	# print(lr.bias)
	# lr.reset()

	# now = time.time_ns()
	# lr.stochastic_gd()
	# then = time.time_ns()
	# print(f'{(then - now)/1e9:.2f}')
	# lr.reset()

	# now = time.time_ns()
	# lr.mini_batch_gd()
	# then = time.time_ns()
	# print(f'{(then - now)/1e9:.2f}')
	# lr.reset()

	# lr.save_to_file()
	# lr.mini_batch_gd()
	# lr.stochastic_gd()
	# print(lr.W)
	# print(lr.bias)

	# pred = lr.predict(data[features])
	# print(classes)
	# print(pred.value_counts())
	# print((pred == target).value_counts())

	# lr.load_weights('weights.csv')
	pred = lr.predict_from_weights(data[features], 'weights.csv')
	# print(classes)
	print(pred)
	print(pred.value_counts())
	correct = (pred == target).value_counts()
	print(correct)
	print(correct.iloc[0] / len(pred))

	# pred = lr.predict_from_weights(scaled_test_df[features], weights)
	# print(pd.DataFrame(pred).value_counts().sort_index())

	# print(pred)


	# xd = []
	# for i, p in enumerate(pred):
	# 	if target.iloc[i] != pred[i]:
	# 		xd.append([int(target.iloc[i]), int(pred[i])])
	# print(pd.DataFrame(xd, columns=['House', 'Guess']).value_counts().sort_values())

if __name__ == "__main__":
	main()
