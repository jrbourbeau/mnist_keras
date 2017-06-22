
from mlxtend.data import mnist_data
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils


def load_train_test(test_size=0.33, random_state=2):

    X, y = mnist_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state, stratify=y)

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test
