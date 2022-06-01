import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json


def main():
    inputs = tf.keras.Input(shape=(30,), name="inputs")
    x = tf.keras.layers.Dense(30, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(26, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    train_ds_input, train_ds_output, output_structure = get_datasets()

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_ds_input, train_ds_output, epochs=1000, batch_size=8000)

    prediction = model.predict(train_ds_input)
    prediction = prediction.tolist()

    nested_prediction = []
    for i in range(len(prediction)):
        nested_prediction.append(tf.nest.pack_sequence_as(output_structure[0], prediction[i]))

    import pprint
    printer = pprint.PrettyPrinter(indent=4)
    printer.pprint(nested_prediction)


def get_datasets():
    directory = os.path.join(os.getcwd(), "datasets")
    all_input_files = np.array([])
    output_structure = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            all_input_files = np.append(all_input_files, filename)
        else:
            continue

    input = []
    output = []
    for i in range(len(all_input_files)):
        x = json.load(open(os.path.join(directory, all_input_files[i])))
        input.append(tf.nest.flatten(x["dataset"]["inputs"]))
        output.append(tf.nest.flatten(x["dataset"]["outputs"]))
        output_structure.append(x["dataset"]["outputs"])

    # show_shapes(input, output)

    tensor_input = tf.convert_to_tensor(input)
    tensor_output = tf.convert_to_tensor(output)

    return tensor_input, tensor_output, output_structure


def show_shapes(Sequences, Targets): # can make yours to take inputs; this'll use local variable values
    print("Expected: (num_samples, timesteps, channels)")
    print("Sequences: {}".format(Sequences.shape))
    print("Targets:   {}".format(Targets.shape))


if __name__ == "__main__":
    main()