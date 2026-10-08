import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "Covid", 1: "Normal", 2: "Viral Pneumonia"}

parser = argparse.ArgumentParser(description="Predict a class from a chest X-ray image.")
parser.add_argument("--image", required=True, help="Path to an X-ray image")
parser.add_argument("--model", default="models/covid_cnn.keras", help="Path to trained Keras model")
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)
img = load_img(args.image, target_size=IMG_SIZE)
arr = img_to_array(img) / 255.0
arr = np.expand_dims(arr, axis=0)

probabilities = model.predict(arr, verbose=0)[0]
index = int(np.argmax(probabilities))

print(f"Predicted class: {CLASS_NAMES[index]}")
print(f"Confidence: {probabilities[index] * 100:.2f}%")
