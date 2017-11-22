import os
import pandas as pd


def load():
    resource_path = os.path.join(
        os.path.abspath('..'),
        'resources')

    paths = [os.path.join(resource_path, p) for p in
             ['train_data.csv', 'train_labels.csv', 'test_data.csv']]

    X, y, test_data = [pd.read_csv(p, header=None)
                       for p in paths]

    X = X.values
    y = y.values.ravel()

    return X, y, test_data
