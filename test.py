import numpy as np

# def pct(list, p):
#     p /= 100
#     list.sort()
#     l = len(list) - 1
#     if (l * p).is_integer():
#         return list[int(l * p)]
#     else:
#         return list[int(l * p)] * (float(l * p) - int(l * p)) + list[int(l * p) + 1] * (1 - (float(l * p) - int(l * p)))
    
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
        list = [0,10,20,30,40,50,60,70,80]

        print(np.percentile(list, 25))
        print(pct(list, 25))
        print(np.percentile(list, 30))
        print(pct(list, 30))
        print(np.percentile(list, 50))
        print(pct(list, 50))
        print(np.percentile(list, 66))
        print(pct(list, 66))
        print(np.percentile(list, 75))
        print(pct(list, 75))

    except Exception as e:
        print(f'{type(e).__name__} : {e}')


if __name__ == "__main__":
    main()
