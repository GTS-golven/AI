import tensorflow as tf
import pandas as pd
import numpy as np
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
    all_input_files = np.array([])
    directory = os.path.join(os.getcwd(), "datasets")

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            all_input_files = np.append(all_input_files, filename)
        else:
            continue

    input_datasets = np.array([[[[]]]])
    output_datasets = np.array([[[]]])

    for i in range(len(all_input_files)):
        dataset = np.array(pd.read_csv(os.path.join(directory, all_input_files[i]), delimiter=";"))
        numpy_input_dataset = np.array([[[]]])
        for j in range(len(dataset)):
            numpy_input_dataset = np.append(numpy_input_dataset, [[dataset[j][4], [dataset[j][0:3]]]])

        input_datasets = np.append(input_datasets, [[numpy_input_dataset]])

        numpy_output_dataset = np.array([[[]]])
        for j in range(len(dataset)):
            numpy_output_dataset = np.append(numpy_output_dataset, [[dataset[j][5:7]], dataset[j][8:9]])
        output_datasets = np.append(output_datasets, [[numpy_output_dataset]])


        print("input datasets \n", input_datasets)
        print("output datasets \n", output_datasets)
        import pdb; pdb.set_trace()

    return input_datasets, output_datasets


if __name__ == "__main__":
    main()
