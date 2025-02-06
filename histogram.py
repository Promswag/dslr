import matplotlib.pyplot as plt
import pandas as pd


def redefine_houses(data):
    house = {c: i for i, c in enumerate(data['Hogwarts House'].unique())}
    data['Hogwarts House'] = data['Hogwarts House'].map(house)
    data = data.select_dtypes(include='number')
    return data


def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        data = redefine_houses(data)
        features = data.iloc[:-1, 2:]

        ravenclaw = data[data['Hogwarts House'] == 0]
        slytherin = data[data['Hogwarts House'] == 1]
        gryffindor = data[data['Hogwarts House'] == 2]
        hufflepuff = data[data['Hogwarts House'] == 3]

        fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(16, 8))
        fig.suptitle("Histogram of the Hogwarts House's courses", fontsize=24)

        for i, feature in enumerate(features):
            ax = axes[int(i / 5), i % 5]
            ax.set_title(feature)
            ax.hist(ravenclaw[feature], bins=30, color='b', alpha=0.4)
            ax.hist(slytherin[feature], bins=30, color='g', alpha=0.4)
            ax.hist(gryffindor[feature], bins=30, color='r', alpha=0.4)
            ax.hist(hufflepuff[feature], bins=30, color='y', alpha=0.4)
            ax.set_xticklabels('')
            ax.set_yticklabels('')

        axes[2, 3].remove()
        axes[2, 4].remove()

        fig.legend(labels=['Ravenclaw', 'Slytherin', 'Gryffindor','Hufflepuff'],
                loc='center', bbox_to_anchor=(0.68, 0.165), fontsize=16)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85)
        plt.savefig('histogram.png')

    except Exception as e:
        print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
    main()

