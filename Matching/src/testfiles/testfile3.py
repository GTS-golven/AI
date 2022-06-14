import numpy as np
import pandas as pd
import tensorflow as tf
import os


def main():
    inputs = tf.keras.Input(shape=(4,), name="inputs")
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(4, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    train_ds_input, train_ds_output = get_datasets()

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds_input, train_ds_output, epochs=1000, batch_size=8000)

    prediction = model.predict(train_ds_input)

    print(prediction)


def get_datasets():
    input = []
    output = []

    all_input_files = np.array([])
    directory = os.path.join(os.getcwd(), "datasets")

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            all_input_files = np.append(all_input_files, filename)
        else:
            continue

    input_datasets = np.ndarray(shape=(len(all_input_files), 6, 2, 4))
    output_datasets = np.ndarray(shape=(len(all_input_files), 8, 2, 3))

    for i in range(len(all_input_files)):
        x = pd.read_json(os.path.join(directory, all_input_files[i]))
        x = x.values.tolist()
        # x = x.to_numpy()
        # x[1][0] = np.asarray(x[1][0], dtype=object)
        # for j in range(len(x[1][0])):
        #     x[1][0][j][1] = np.asarray(x[1][0][j][1], dtype=object)
        # x[2][0] = np.asarray(x[2][0], dtype=object)
        # for j in range(len(x[2][0])):
        #     x[2][0][i] = np.asarray(x[2][0][i], dtype=object)
        #     if not j == 6 and not j == 7:
        #         x[2][0][j] = np.asarray(x[2][0][j], dtype=object)
        #         x[2][0][j][1] = np.asarray(x[2][0][j][1], dtype=object)

        input = np.append(input, [x[1]])
        output = np.append(output, [x[2]])

    show_shapes(input, output)

    input = np.asarray(input).astype('float32')
    input = tf.convert_to_tensor(input)

    # input = convert_numpy_array_to_tensor(input)
    # output = convert_numpy_array_to_tensor(output)

    return input, output




# def convert_numpy_array_to_tensor(item):
#     for i in range(len(item)):
#         if type(item[i]) == np.ndarray:
#             item[i] = convert_numpy_array_to_tensor(item[i])
#         print(item[i])
#         if type(item[i]) is not np.ndarray:
#             return item[i]
#             # return tf.convert_to_tensor(item[i])
#         return item[i]
#         z = item[i].dtype
#         x = tf.as_dtype(item[i].dtype)
#         # return item[i]
#         # return tf.convert_to_tensor(np.asarray(item[i], dtype=tf.as_dtype(item[i].dtype)))
#         return tf.convert_to_tensor()


def show_shapes(Sequences, Targets): # can make yours to take inputs; this'll use local variable values
    print("Expected: (num_samples, timesteps, channels)")
    print("Sequences: {}".format(Sequences.shape))
    print("Targets:   {}".format(Targets.shape))
    print(f"ndim: {Sequences.ndim}")


if __name__ == "__main__":
    main()