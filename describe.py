import pandas as pd
import numpy as np
import sys


def pct(list, p):
    if p < 0 or p > 100:
        raise ValueError("parameter p must be >= 0 and <= 100")
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
        data = pd.read_csv(sys.argv[1])
        data = data.select_dtypes(include=[np.number])
        data = data.drop(labels='Index', axis=1)

        count = {str: int}
        mean_ = {str: int}
        std__ = {str: int}
        mini_ = {str: int}
        pct25 = {str: int}
        pct50 = {str: int}
        pct75 = {str: int}
        maxi_ = {str: int}
        rng__ = {str: int}
        miss_ = {str: int}
        skew_ = {str: int}
        kurt_ = {str: int}
        kurtx = {str: int}
        pct69 = {str: int}

        features = data.columns[1:]

        for f in features:
            df = data[f].dropna().sort_values(ascending=True).reset_index(drop=True)
            count[f] = len(df)
            mean_[f] = sum(df) / count[f]
            std__[f] = (sum(abs(df - mean_[f]) ** 2) / count[f]) ** 0.5
            mini_[f] = df[0]
            pct25[f] = pct(df.values, 25)
            pct50[f] = pct(df.values, 50)
            pct75[f] = pct(df.values, 75)
            maxi_[f] = df[count[f] - 1]
            rng__[f] = maxi_[f] - mini_[f] if count[f] > 0 else np.nan
            miss_[f] = data[f].isna().sum()
            skew_[f] = (1 / count[f]) * sum(((df - mean_[f]) / std__[f]) ** 3)
            kurt_[f] = (1 / count[f]) * sum(((df - mean_[f]) / std__[f]) ** 4)
            kurtx[f] = kurt_[f] - 3
            pct69[f] = pct(df.values, 69)

        print(''.ljust(10) + ' '.join(f'{f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('count'.ljust(10) + ' '.join(f'{count[f]:.0f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('mean '.ljust(10) + ' '.join(f'{mean_[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('std  '.ljust(10) + ' '.join(f'{std__[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('min  '.ljust(10) + ' '.join(f'{mini_[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('25%  '.ljust(10) + ' '.join(f'{pct25[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('50%  '.ljust(10) + ' '.join(f'{pct50[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('75%  '.ljust(10) + ' '.join(f'{pct75[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('max  '.ljust(10) + ' '.join(f'{maxi_[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('\nbonus:')
        print('range'.ljust(10) + ' '.join(f'{rng__[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('miss '.ljust(10) + ' '.join(f'{miss_[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('skew_'.ljust(10) + ' '.join(f'{skew_[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('kurt_'.ljust(10) + ' '.join(f'{kurt_[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('kurtX'.ljust(10) + ' '.join(f'{kurtx[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))
        print('75%  '.ljust(10) + ' '.join(f'{pct69[f]:.3f}'.rjust(len(f)+1 if len(f) > 9 else 10) for f in features))

    except Exception as e:
        print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
    main()
