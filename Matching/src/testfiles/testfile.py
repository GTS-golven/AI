from sklearn.datasets import load_iris
from tensorflow import keras
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

from src.encode import *


def get_datasets():
    # TODO: Implement this function using the real dataset
    file_url = "datasets/Mock_real_dataset.csv"
    dataframe = pd.read_csv(file_url, delimiter=";")

    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)

    print(
        "Using %d samples for training and %d for validation"
        % (len(train_dataframe), len(val_dataframe))
    )

    train_ds = np.array(train_dataframe)
    val_ds = np.array(val_dataframe)

    return train_ds, val_ds


# Inputs
X1 = keras.Input(shape=(1,), name="X1")
X2 = keras.Input(shape=(1,), name="X2")
Y1 = keras.Input(shape=(1,), name="Y1")
Y2 = keras.Input(shape=(1,), name="Y2")
TInput = keras.Input(shape=(1,), name="TInput")

# Outputs
X_out = keras.Input(shape=(1,), name="X_out")
Y_out = keras.Input(shape=(1,), name="Y_out")
Z_out = keras.Input(shape=(1,), name="Z_out")
Start_angle_out = keras.Input(shape=(1,), name="Start_angle_out")
Speed_out = keras.Input(shape=(1,), name="Speed_out")

all_inputs = [
    X1,
    X2,
    Y1,
    Y2,
    TInput
]
print(all_inputs)
all_outputs = [
    X_out,
    Y_out,
    Z_out,
    Start_angle_out,
    Speed_out
]

train_ds, val_ds = get_datasets()

vectorizer = layers.experimental.preprocessing.TextVectorization(output_mode="int")

vectorizer.adapt(train_ds)

integer_data = vectorizer(train_ds)
print(integer_data)






inputs = Input(shape=(5,), name="inputs")
hidden1 = Dense(25, activation='relu')(inputs)
hidden2 = Dense(2500, activation='relu')(hidden1)
output1 = Dense(5)(hidden2)

model = Model(inputs=inputs, outputs=output1)

model.compile(loss=['mae', 'sparse_categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
print(all_inputs)
print(all_outputs)

history = model.fit(all_inputs, all_outputs, epochs=10, batch_size=8)
