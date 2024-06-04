import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class_names = ["bird", "cat", "deer", "dog"]

model = load_model("model.keras")

def classify(input_img):
    # We need to "normalize" the input.
    # Input pixels are between 0 and 255,
    # but neural net expects values 0 to 1.
    input_img = np.array(input_img) / 255

    # Add a batch dimension of size 1.
    input_img = np.array([input_img])

    # Run our image through the model.
    prediction = model.predict(input_img)

    # Remove batch dimension from output.
    prediction = prediction[0]

    # Turn softmax output into index.
    prediction = np.argmax(prediction)

    # Turn index into class name
    return class_names[prediction]

demo = gr.Interface(classify, gr.Image(), "text")
demo.launch()
