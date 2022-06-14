import numpy as np
import pandas as pd
import tensorflow as tf
import os


def get_datasets():
    all_input_files = np.array([])
    directory = os.path.join(os.getcwd(), "datasets")

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            all_input_files = np.append(all_input_files, filename)
        else:
            continue

    for i in range(len(all_input_files)):
        x = pd.read_json(os.path.join(directory, all_input_files[i]))
        x = x.to_numpy()
        x[1][0] = np.asarray(x[1][0])
        for j in range(len(x[1][0])):
            x[1][0][j][1] = np.asarray(x[1][0][j][1])
        x[2][0] = np.asarray(x[2][0])
        for j in range(len(x[2][0])):
            x[2][0][i] = np.asarray(x[2][0][i])
            if not j == 6 and not j == 7:
                x[2][0][j][1] = np.asarray(x[2][0][j][1])

        y = convert_numpy_array_to_tensor(x[1])
        for j in range(len(x)):
            if not j == 0:
                y = convert_numpy_array_to_tensor(x[j])
                print(y)


def convert_numpy_array_to_tensor(item):
    for i in range(len(item)):
        if type(item[i]) == np.ndarray:
            item[i] = convert_numpy_array_to_tensor(item[i])
        if type(item[i]) is not np.ndarray:
            return item[i]
        return tf.convert_to_tensor(np.asarray(item[i]).astype(np.int))


if __name__ == "__main__":
    get_datasets()