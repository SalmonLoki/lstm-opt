import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2):
    mnist = fetch_openml('MNIST original')
    X_train, X_val, y_train, y_val = train_test_split(mnist.data, mnist.target, test_size=test_size)

    X_train = X_train.astype(np.float32) / 255.
    X_val = X_val.astype(np.float32) / 255.

    return X_train, y_train, X_val, y_val


def iterate_minibatches(*arrays, **kwargs):
    batch_size = kwargs.get("batch_size", 100)
    shuffle = kwargs.get("shuffle", True)

    if shuffle:
        indices = np.arange(len(arrays[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(arrays[0]) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        yield [arr[excerpt] for arr in arrays]