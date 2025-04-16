# siamese_train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.datasets import fetch_lfw_pairs
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import cv2

# Define base model
def build_base_model():
    inputs = layers.Input(shape=(100, 100, 1))
    x = layers.Conv2D(64, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)  # 👈 normalize
    return Model(inputs, x, name="base_model")


def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)
    return loss

# Only run training if this script is executed directly
if __name__ == "__main__":
    os.makedirs("training_logs", exist_ok=True)

    # Load LFW pairs
    lfw = fetch_lfw_pairs(subset='train', color=True, resize=0.5)
    X, y = lfw.pairs, lfw.target

    # Convert to grayscale and resize to 100x100
    X_gray = np.array([cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_RGB2GRAY) for pair in X for img in pair])
    X_gray = X_gray.reshape(-1, 2, 100, 100, 1) / 255.0

    X1 = X_gray[:, 0, :, :, :]
    X2 = X_gray[:, 1, :, :, :]

    y = np.array(y).astype('float32')

    # Split
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

    # Build model
    base_model = build_base_model()
    input_a = layers.Input(shape=(100, 100, 1))
    input_b = layers.Input(shape=(100, 100, 1))

    feat_a = base_model(input_a)
    feat_b = base_model(input_b)

    distance = layers.Lambda(euclidean_distance)([feat_a, feat_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)

    params = {
        "optimizer": "adam",
        "margin": 1.0,
        "batch_size": 32,
        "epochs": 30
    }

    model.compile(loss=contrastive_loss(params["margin"]), optimizer=params["optimizer"])

    # Train model
    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        batch_size=params["batch_size"],
        epochs=params["epochs"]
    )

    # Save model
    base_model.save("base_siamese_model.h5")

    # Save plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Loss over Epochs')
    plt.legend()
    plt.savefig("training_logs/loss_curve.png")

    # Save training info
    with open("training_logs/params.json", "w") as f:
        json.dump(params, f, indent=4)

    np.save("training_logs/history.npy", history.history)

    log_df = pd.DataFrame(history.history)
    log_df.index.name = 'epoch'
    log_df.to_csv("training_logs/training_log.csv")
