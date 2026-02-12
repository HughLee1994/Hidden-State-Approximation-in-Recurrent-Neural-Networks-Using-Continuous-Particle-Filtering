import numpy as np
import pandas as pd


def read_data(input_path, debug=True):
    """
    read the dataset
    :param input_path: directory
    :param debug:
    :return: x: features
             y: ground truth
    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].to_numpy()
    y = np.array(df.NDX)
    print("driving series size:", X.shape)
    print("predict series siez:", y.shape)
    return X, y