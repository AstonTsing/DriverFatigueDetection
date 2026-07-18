"""对比 CPU(TF2.21) 与 DirectML GPU(TF2.10) 的单步训练耗时。"""
import os
import sys
import time

import numpy as np

BATCH = 32
IMG = (224, 224, 3)
STEPS = 15
WARMUP = 3


def bench_env(label: str, use_dml: bool):
    if use_dml:
        os.environ.setdefault("DML_VISIBLE_DEVICES", "1")
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.models import Sequential

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n=== {label} | TF {tf.__version__} | GPU={gpus} ===")

    base = VGG16(include_top=False, weights=None, input_shape=IMG)
    base.trainable = False
    model = Sequential([
        base,
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x = np.random.rand(BATCH, *IMG).astype(np.float32)
    y = np.random.randint(0, 2, size=(BATCH, 1)).astype(np.float32)

    for _ in range(WARMUP):
        model.train_on_batch(x, y)

    t0 = time.perf_counter()
    for _ in range(STEPS):
        model.train_on_batch(x, y)
    elapsed = time.perf_counter() - t0
    per_step = elapsed / STEPS
    print(f"train_on_batch: {per_step*1000:.1f} ms/step ({STEPS} steps)")
    return per_step


def main():
    py = sys.executable
    if "drowsiness-gpu" in py:
        bench_env("DirectML GPU env", use_dml=True)
    else:
        bench_env("CPU env", use_dml=False)


if __name__ == "__main__":
    main()
