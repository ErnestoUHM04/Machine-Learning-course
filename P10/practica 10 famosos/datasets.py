# 1_datasets.py

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from config import *

def celeba_to_example(feat):
    img = tf.image.resize(feat["image"], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1  # preprocess_input manual
    label = 1
    return img, tf.constant(label, tf.int32)

def lfw_to_example(feat):
    img = tf.image.resize(feat["image"], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1
    label = 0
    return img, tf.constant(label, tf.int32)

print("Cargando datasets...")

celeb_train = tfds.load("celeb_a", split=CELEB_TRAIN, as_supervised=False)
celeb_val   = tfds.load("celeb_a", split=CELEB_VAL, as_supervised=False)
celeb_test  = tfds.load("celeb_a", split=CELEB_TEST, as_supervised=False)

lfw_train = tfds.load("lfw", split=LFW_TRAIN, as_supervised=False)
lfw_val   = tfds.load("lfw", split=LFW_VAL, as_supervised=False)
lfw_test  = tfds.load("lfw", split=LFW_TEST, as_supervised=False)

def prepare(celeb, lfw):
    ds = celeb.map(celeba_to_example).concatenate(
         lfw.map(lfw_to_example)
    )
    ds = ds.shuffle(10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = prepare(celeb_train, lfw_train)
val_ds   = prepare(celeb_val, lfw_val)
test_ds  = prepare(celeb_test, lfw_test)

print("Datasets listos.")