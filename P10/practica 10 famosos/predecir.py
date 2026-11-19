# 4_predecir.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image as keras_image
from config import *

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_DIR)

def predecir():
    root = tk.Tk()
    root.withdraw()
    fp = filedialog.askopenfilename(filetypes=[("IMG", "*.jpg *.jpeg *.png")])
    if not fp:
        return

    img = keras_image.load_img(fp, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img)
    arr = (arr / 127.5) - 1
    batch = np.expand_dims(arr, 0)

    prob = float(model.predict(batch)[0][0])
    label = "Famoso" if prob > 0.5 else "No Famoso"

    print(f"\n{fp}")
    print(f"Probabilidad: {prob*100:.2f}%  -> {label}")

    plt.imshow(arr/2 + 0.5)
    plt.title(f"{label} ({prob*100:.1f}%)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    predecir()