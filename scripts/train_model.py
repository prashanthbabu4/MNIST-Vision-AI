"""
Model Training Script for MNIST Digits Classification
Replicates the exact training pipeline from 'MNIST Digits Classification [0 to 9].ipynb'
and saves the trained model to models/digits_intel.h5 (and models/mnist_model.h5).
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Ensure Windows DLL search directory is configured
BASE_DIR = Path(__file__).resolve().parent.parent
venv_scripts = BASE_DIR / ".venv" / "Scripts"
if venv_scripts.exists() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(venv_scripts))
    except Exception:
        pass
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "digits_intel.h5"
MODEL_ALIAS_PATH = MODELS_DIR / "mnist_model.h5"

print("=" * 60)
print("MNIST Vision AI — Model Training Pipeline")
print("=" * 60)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense

print("[1/5] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f"      Training data shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"      Testing data shape:  {X_test.shape}, Labels: {y_test.shape}")

print("[2/5] Normalizing pixel values to [0.0, 1.0]...")
X_train = X_train / 255.0
X_test = X_test / 255.0

print("[3/5] Defining Model Architecture (per notebook specification)...")
# Flatten() -> Dense(100, relu) -> Dense(30, relu) -> Dense(10, softmax)
model = Sequential([
    Flatten(input_shape=(28, 28), name="flatten_input"),
    Dense(units=100, activation="relu", name="dense_100"),
    Dense(units=30, activation="relu", name="dense_30"),
    Dense(units=10, activation="softmax", name="output_digits")
])

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

print("[4/5] Training model for 10 epochs (batch_size=32)...")
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate on test set
eval_res = model.evaluate(X_test, y_test, verbose=0)
test_loss = eval_res[0]
test_acc = eval_res[1] if len(eval_res) > 1 else None
print(f"\nFinal Test Loss: {test_loss:.4f}" + (f", Test Accuracy: {test_acc * 100:.2f}%" if test_acc else ""))

print(f"[5/5] Saving model file to '{MODEL_OUTPUT_PATH}'...")
model.save(str(MODEL_OUTPUT_PATH))

# Also save alias copy
import shutil
shutil.copy2(MODEL_OUTPUT_PATH, MODEL_ALIAS_PATH)
print(f"      Created alias at '{MODEL_ALIAS_PATH}'")

print("\n Model training and serialization completed successfully!")
