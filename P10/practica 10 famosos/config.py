# 0_config.py

import tensorflow as tf

# Tamaño de imagen para MobileNetV2
IMG_SIZE = 160
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 8
LEARNING_RATE = 1e-4

MODEL_DIR = "saved_model/famous_detector_mobilenetv2"

# Particiones de datasets
CELEB_TRAIN = "train[:8000]"
CELEB_VAL   = "train[8000:9000]"
CELEB_TEST  = "test[:1000]"

LFW_TRAIN = "train[:8000]"
LFW_VAL   = "train[8000:9000]"
LFW_TEST  = "train[9000:10000]"