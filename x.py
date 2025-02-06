import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import preprocessing, standardize, LogisticRegression

def compute_accuracy(predictions, targets):
    return np.mean(predictions == targets) * 100

def main():
    df = pd.read_csv("datasets/dataset_train.csv")
    df = preprocessing(df, "Hogwarts House", keep_na=True)
    features = df.iloc[:, 2:-1]  # Excluding target column
    data = pd.DataFrame()

    for f in features:
        data[f] = standardize(df[f])
    target = df["Hogwarts House"]

    print("Standardized Features:\n", data.head(), '\n')

    # print("Gradient Descent Training:")
    # lr_gd = LogisticRegression(data, target, learning_rate=0.01, epochs=1000)
    # lr_gd.gradient_descent()
    # print("Weights (Gradient Descent):\n", lr_gd.W)
    # print("Bias (Gradient Descent):\n", lr_gd.bias)

    # pred_gd = lr_gd.predict(data)
    # accuracy_gd = compute_accuracy(pred_gd, target)
    # print("\nGradient Descent Accuracy:", accuracy_gd, "%")

    print("\nAdam Training:")
    lr_adam = LogisticRegression(data, target, learning_rate=0.01, epochs=1000)
    lr_adam.adam()
    print("Weights (Adam):\n", lr_adam.W)
    print("Bias (Adam):\n", lr_adam.bias)

    pred_adam = lr_adam.predict(data)
    accuracy_adam = compute_accuracy(pred_adam, target)
    print("\nAdam Accuracy:", accuracy_adam, "%")

    # Plot Adam path in 3D
    lr_adam.plot_adam_path()

if __name__ == "__main__":
    main()


