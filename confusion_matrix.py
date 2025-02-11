import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os


def main():
    try:
        real_houses = pd.read_csv('datasets/dataset_train.csv')['Hogwarts House']
        predicted_houses = pd.read_csv(f'datasets/{sys.argv[1]}.csv')['Hogwarts House']

        unique_houses = sorted(real_houses.unique())
        cm = confusion_matrix(real_houses, predicted_houses, labels=unique_houses)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_houses)
        disp.plot(cmap=plt.cm.Blues)

        plt.title("Confusion Matrix")

        os.makedirs("graphs", exist_ok=True)
        plt.savefig('graphs/confusion_matrix.png')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
