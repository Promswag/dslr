import pandas as pd
import numpy as np


def pct(list, p):
    list.sort()
    true_index = (p / 100) * (len(list) - 1)
    index = int(true_index)
    ratio = true_index - index

    if ratio == 0:
        return list[index]
    else:
        base = list[index]
        diff = list[index + 1] - list[index]
        return base + ratio * diff


def main():
    try:
        data = pd.read_csv("dataset_train.csv")
        data = data.select_dtypes(include=[np.number])

        count = {str: int}
        mean = {str: int}
        std = {str: int}
        mini = {str: int}
        pct25 = {str: int}
        pct50 = {str: int}
        pct75 = {str: int}
        maxi = {str: int}

        features = data.columns[1:]

        for f in features:
            df = data[f].dropna().sort_values(ascending=True).reset_index(drop=True)
            count[f] = len(df)
            mean[f] = sum(df) / count[f]
            std[f] = (sum(abs(df - mean[f]) ** 2) / count[f]) ** 0.5
            mini[f] = df[0]
            pct25[f] = pct(df.values, 25)
            pct50[f] = pct(df.values, 50)
            pct75[f] = pct(df.values, 75)
            maxi[f] = df[count[f] - 1]

        print(data.describe())

    except Exception as e:
        print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
    main()
