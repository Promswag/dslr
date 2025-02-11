import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression, fill_na
from StandardScaler import StandardScaler
from sklearn.metrics import accuracy_score

def compute_accuracy(predictions, targets):
    return np.mean(predictions == targets) * 100

def main():
    df = pd.read_csv("datasets/dataset_train.csv")
    features = [
		'Defense Against the Dark Arts',
		'Ancient Runes',
		'Charms',
	]
    target = 'Hogwarts House'
    df = fill_na(df, target)
    # features = df.iloc[:, 2:-1]  # Excluding target column
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    print("Standardized Features:\n", df.head(), '\n')

    # print("Gradient Descent Training:")
    # lr_gd = LogisticRegression(data, target, learning_rate=0.01, epochs=1000)
    # lr_gd.gradient_descent()
    # print("Weights (Gradient Descent):\n", lr_gd.W)
    # print("Bias (Gradient Descent):\n", lr_gd.bias)

    # pred_gd = lr_gd.predict(data)
    # accuracy_gd = compute_accuracy(pred_gd, target)
    # print("\nGradient Descent Accuracy:", accuracy_gd, "%")

    print("\nAdam Training:")
    lr_adam = LogisticRegression(df[features], df[target], learning_rate=0.01, epochs=100)
    lr_adam.adam()
    print("Weights (Adam):\n", lr_adam.W)
    print("Bias (Adam):\n", lr_adam.bias)

    pred_adam = lr_adam.predict(df[features])
    accuracy_adam = compute_accuracy(pred_adam, target)
    print("\nAdam Accuracy:", accuracy_adam, "%")

    # Plot Adam path in 3D
    # lr_adam.plot_adam_path()
    
    # correct = (pred_adam == df[target]).value_counts()
    # print(correct, correct[0] / len(df[target]))
    print(accuracy_score(pred_adam, df[target]))

if __name__ == "__main__":
    main()


