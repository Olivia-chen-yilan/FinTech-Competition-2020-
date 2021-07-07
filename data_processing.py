import numpy as np
def shuffle_split_data(x, y, percent):
    arr_rand = np.random.rand(x.shape[0])
    split = arr_rand < np.percentile(arr_rand, percent)

    x_train = x[split]
    y_train = y[split]
    x_test =  x[~split]
    y_test = y[~split]

    print (len(x_train), len(y_train), len(x_test), len(y_test))
    return x_train, y_train, x_test, y_test