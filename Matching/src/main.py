import os

import pandas as pd
from tensorflow.keras import layers
from src.encode import *
from matplotlib import pyplot as plt

import pprint

my_printer = pprint.PrettyPrinter(indent=4)


def main():
    model = get_model()

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    history = model.fit(train_ds, epochs=1000, validation_data=val_ds, batch_size=1)

    model.save('model')

    get_plot(history)


def get_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def get_dataset():
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    dataframe = pd.read_csv(file_url)

    val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
    train_dataframe = dataframe.drop(val_dataframe.index)

    print(
        "Using %d samples for training and %d for validation"
        % (len(train_dataframe), len(val_dataframe))
    )

    global train_ds
    global val_ds

    train_ds = dataframe_to_dataset(train_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)


def get_model():
    if os.path.isfile('model'):
        return keras.models.load_model('model')

    else:
        # Categorical features encoded as integers
        sex = keras.Input(shape=(1,), name="sex", dtype="int64")
        cp = keras.Input(shape=(1,), name="cp", dtype="int64")
        fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
        restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
        exang = keras.Input(shape=(1,), name="exang", dtype="int64")
        ca = keras.Input(shape=(1,), name="ca", dtype="int64")

        # Categorical feature encoded as string
        thal = keras.Input(shape=(1,), name="thal", dtype="string")

        # Numerical features
        age = keras.Input(shape=(1,), name="age")
        trestbps = keras.Input(shape=(1,), name="trestbps")
        chol = keras.Input(shape=(1,), name="chol")
        thalach = keras.Input(shape=(1,), name="thalach")
        oldpeak = keras.Input(shape=(1,), name="oldpeak")
        slope = keras.Input(shape=(1,), name="slope")

        all_inputs = [
            sex,
            cp,
            fbs,
            restecg,
            exang,
            ca,
            thal,
            age,
            trestbps,
            chol,
            thalach,
            oldpeak,
            slope,
        ]

        # Integer categorical features
        sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
        cp_encoded = encode_categorical_feature(cp, "cp", train_ds, False)
        fbs_encoded = encode_categorical_feature(fbs, "fbs", train_ds, False)
        restecg_encoded = encode_categorical_feature(restecg, "restecg", train_ds, False)
        exang_encoded = encode_categorical_feature(exang, "exang", train_ds, False)
        ca_encoded = encode_categorical_feature(ca, "ca", train_ds, False)

        # String categorical features
        thal_encoded = encode_categorical_feature(thal, "thal", train_ds, True)

        # Numerical features
        age_encoded = encode_numerical_feature(age, "age", train_ds)
        trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
        chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
        thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
        oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
        slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

        all_features = layers.concatenate(
            [
                sex_encoded,
                cp_encoded,
                fbs_encoded,
                restecg_encoded,
                exang_encoded,
                slope_encoded,
                ca_encoded,
                thal_encoded,
                age_encoded,
                trestbps_encoded,
                chol_encoded,
                thalach_encoded,
                oldpeak_encoded,
            ]
        )
        x = layers.Dense(32, activation="relu")(all_features)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(all_inputs, output)
    return model


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    print(dataframe)
    print(labels)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def sample():
    sample = {
        "age": 60,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": "fixed",
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}

    model = get_model()
    predictions = model.predict(input_dict)

    print(predictions)


def test():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


if __name__ == '__main__':
    get_dataset()

    main()
    # sample()
    test()
