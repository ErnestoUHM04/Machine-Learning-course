#  pip install "tensorflow[and-cuda]" tensorflow-datasets matplotlib numpy pillow keras beautifulsoup4




import os
import sys
import subprocess
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import tkinter as tk
from tkinter import filedialog

# intentar instalar bs4 automáticamente si falta (TFDS lo usa para Google Drive)
try:
    import bs4  # noqa: F401
except Exception:
    print("bs4 (beautifulsoup4) no instalado. Intentando instalarlo automáticamente...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
        import bs4  # noqa: F401
        print("bs4 instalado correctamente.")
    except Exception:
        print("No se pudo instalar bs4 automáticamente. Ejecuta: pip install beautifulsoup4")
        raise

# --------------------------
# Config
# --------------------------
IMG_SIZE = 160                # tamaño para MobileNetV2 (160 o 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 8                    # puedes aumentar
LEARNING_RATE = 1e-4
MODEL_DIR = "saved_model/famous_detector_mobilenetv2"

# Ajusta estos slices para controlar cuántas imágenes descargar/usar
CELEB_TRAIN_SPLIT = "train[:8000]"
CELEB_VAL_SPLIT   = "train[8000:9000]"
CELEB_TEST_SPLIT  = "test[:1000]"

LFW_TRAIN_SPLIT = "train[:8000]"
LFW_VAL_SPLIT   = "train[8000:9000]"
LFW_TEST_SPLIT  = "train[9000:10000]"

print("TensorFlow:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))

# --------------------------
# Util: convertir features TFDS -> imagen + etiqueta binaria
# --------------------------
def celeba_to_example(feat):
    # CelebA -> clase 1 (famoso)
    img = tf.image.resize(feat["image"], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    # preprocess_input espera rango [-1,1]
    img = preprocess_input(img)
    label = tf.constant(1, dtype=tf.int32)
    return img, label

def lfw_to_example(feat):
    # LFW -> clase 0 (no famoso)
    img = tf.image.resize(feat["image"], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label = tf.constant(0, dtype=tf.int32)
    return img, label

# --------------------------
# Cargar datasets (TFDS)
# --------------------------
print("Descargando / cargando datasets (CelebA y LFW). Esto puede tardar...")
ds_celeb_train_raw = tfds.load("celeb_a", split=CELEB_TRAIN_SPLIT, as_supervised=False)
ds_celeb_val_raw   = tfds.load("celeb_a", split=CELEB_VAL_SPLIT, as_supervised=False)
ds_celeb_test_raw  = tfds.load("celeb_a", split=CELEB_TEST_SPLIT, as_supervised=False)

ds_lfw_train_raw = tfds.load("lfw", split=LFW_TRAIN_SPLIT, as_supervised=False)
ds_lfw_val_raw   = tfds.load("lfw", split=LFW_VAL_SPLIT, as_supervised=False)
ds_lfw_test_raw  = tfds.load("lfw", split=LFW_TEST_SPLIT, as_supervised=False)

# --------------------------
# Mapear y preparar pipelines
# --------------------------
ds_celeb_train = ds_celeb_train_raw.map(celeba_to_example, num_parallel_calls=AUTOTUNE)
ds_celeb_val   = ds_celeb_val_raw.map(celeba_to_example, num_parallel_calls=AUTOTUNE)
ds_celeb_test  = ds_celeb_test_raw.map(celeba_to_example, num_parallel_calls=AUTOTUNE)

ds_lfw_train = ds_lfw_train_raw.map(lfw_to_example, num_parallel_calls=AUTOTUNE)
ds_lfw_val   = ds_lfw_val_raw.map(lfw_to_example, num_parallel_calls=AUTOTUNE)
ds_lfw_test  = ds_lfw_test_raw.map(lfw_to_example, num_parallel_calls=AUTOTUNE)

# Combinar positivos y negativos
def combine_and_prepare(celeb_ds, lfw_ds, shuffle_buffer=10000):
    ds = celeb_ds.concatenate(lfw_ds)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = combine_and_prepare(ds_celeb_train, ds_lfw_train)
val_ds   = combine_and_prepare(ds_celeb_val, ds_lfw_val)
test_ds  = combine_and_prepare(ds_celeb_test, ds_lfw_test)

# Sanity check
for imgs, labs in train_ds.take(1):
    print("Batch shape:", imgs.shape, " Labels shape:", labs.shape)
    print("Labels example (counts):", np.bincount(labs.numpy().astype(int)))

# --------------------------
# Modelo: MobileNetV2 + cabeza binaria
# --------------------------
def build_famous_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    base.trainable = False  # freeze inicialmente

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_famous_model()
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auroc")])
model.summary()

# Callbacks
os.makedirs(MODEL_DIR, exist_ok=True)
ckpt_cb = callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best.ckpt"),
                                    save_weights_only=True, save_best_only=True,
                                    monitor="val_auroc", mode="max", verbose=1)
es_cb = callbacks.EarlyStopping(monitor="val_auroc", patience=3, restore_best_weights=True, mode="max", verbose=1)

# --------------------------
# Entrenamiento (fase 1: cabeza)
# --------------------------
print("Entrenamiento fase 1: entrenar la cabeza (base congelada)")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt_cb, es_cb])

# --------------------------
# Fine-tuning (fase 2): descongelar algunas capas del base
# --------------------------
print("Fine-tuning: descongelando últimas capas del backbone")
base = model.layers[1] if isinstance(model.layers[1], MobileNetV2) else None
# más robusto: localizar la capa base por tipo
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
        base = layer
        break

if base is None:
    print("No se pudo identificar el backbone; omitiendo fine-tune.")
else:
    base.trainable = True
    # descongelar últimas N capas (ejemplo últimas 30)
    fine_tune_at = len(base.layers) - 30
    for i, l in enumerate(base.layers):
        l.trainable = i >= fine_tune_at

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auroc")])

    history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt_cb, es_cb])

# --------------------------
# Evaluación final
# --------------------------
print("Evaluando en test set...")
res = model.evaluate(test_ds)
print("Test (loss, acc, auroc):", res)

# Guardar modelo completo
model.save(MODEL_DIR)
print("Modelo guardado en:", MODEL_DIR)

# --------------------------
# Función de predicción interactiva (Tkinter)
# --------------------------
def predict_file(model_to_use, threshold=0.5):
    root = tk.Tk()
    root.withdraw()
    print("\nSelecciona una imagen (jpg/png)...")
    fp = filedialog.askopenfilename(title="Selecciona imagen", filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not fp:
        print("No seleccionado.")
        return
    img = keras_image.load_img(fp, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img)
    arr = preprocess_input(arr)
    batch = np.expand_dims(arr, axis=0)
    p = model_to_use.predict(batch)[0][0]
    label = "Famoso" if p >= threshold else "No Famoso"
    print(f"\nArchivo: {fp}\n Probabilidad Famoso: {p*100:.2f}%  -> {label}")
    plt.imshow((keras_image.img_to_array(img)/255.0))
    plt.axis("off")
    plt.title(f"{label} ({p*100:.1f}%)")
    plt.show()

# Llamar predict interactivo (descomenta si quieres ejecutar inmediatamente)
if __name__ == "__main__":
    predict_file(model)