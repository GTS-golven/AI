import tensorflow as tf
import os
import json

def predict(input):
    # Make array of input
    data = []
    data.append(tf.nest.flatten(input))
    
    # Load the model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "./models/model.h5"))
    print("Model loaded")
    
    # Make a prediction
    prediction = model.predict(tf.convert_to_tensor(data))
    prediction = prediction.tolist()
    return tf.nest.pack_sequence_as(get_structure(), prediction[0])

def get_structure():
    # Check for json files in datasets directory and return outputs for structure
    directory = os.path.join(os.getcwd(), "datasets")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            x = json.load(open(os.path.join(directory, filename)))
            return x["outputs"]
        else:
            continue