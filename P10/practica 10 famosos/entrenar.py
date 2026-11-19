# 3_entrenar.py

import os
import tensorflow as tf
from tensorflow.keras import callbacks
from config import *
from datasets import train_ds, val_ds, test_ds
from modelo import build_model

os.makedirs(MODEL_DIR, exist_ok=True)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auroc")]
)

ckpt = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best.ckpt"),
    save_weights_only=True,
    save_best_only=True,
    monitor="val_auroc",
    mode="max"
)

print("Entrenando modelo...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt])

print("Cargando mejores pesos...")
model.load_weights(os.path.join(MODEL_DIR, "best.ckpt"))

print("Evaluando...")
print(model.evaluate(test_ds))

model.save(MODEL_DIR)
print("Modelo guardado.")