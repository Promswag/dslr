import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys


def main():
    try:
        real = pd.read_csv('datasets/dataset_train.csv')['Hogwarts House']
        predictions = pd.read_csv(f'datasets/{sys.argv[1]}.csv')['Hogwarts House']

        cm = confusion_matrix(real, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)

        plt.title("Confusion Matrix")
        plt.savefig('confusion_matrix.png')

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
