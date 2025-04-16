# siamese_train_model.py (Minimal Training Version)

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
import random
import copy

# ----- Base Model (Lightweight) -----
def build_base_model():
    inputs = layers.Input(shape=(100, 100, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return Model(inputs, x, name="base_model")

def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-9)

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)
    return loss

# ----- Load and Preprocess Dataset (with sample limit) -----
def load_dataset(max_samples=1000):
    lfw = fetch_lfw_pairs(subset='train', color=True, resize=0.5)
    X, y = lfw.pairs, lfw.target

    # Limit dataset
    if max_samples and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[indices], y[indices]

    X_gray = np.array([cv2.cvtColor(cv2.resize(img, (100, 100)), cv2.COLOR_RGB2GRAY)
                       for pair in X for img in pair])
    X_gray = X_gray.reshape(-1, 2, 100, 100, 1).astype('float32') / 255.0
    X1, X2 = X_gray[:, 0, :, :, :], X_gray[:, 1, :, :, :]
    return train_test_split(X1, X2, y.astype('float32'), test_size=0.2, random_state=42)

# ----- Fitness Evaluation -----
def train_and_evaluate(params, X1_train, X2_train, y_train, X1_val, X2_val, y_val):
    tf.keras.backend.clear_session()
    base_model = build_base_model()
    input_a = layers.Input(shape=(100, 100, 1))
    input_b = layers.Input(shape=(100, 100, 1))
    feat_a, feat_b = base_model(input_a), base_model(input_b)
    distance = layers.Lambda(euclidean_distance)([feat_a, feat_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    
    model.compile(loss=contrastive_loss(params["margin"]),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]))

    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_val, X2_val], y_val),
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=0
    )
    final_val_loss = history.history['val_loss'][-1]
    return final_val_loss, model, base_model, history

# ----- Evolutionary Algorithm -----
def initialize_population(size):
    return [
        {
            "batch_size": random.choice([16, 32]),
            "learning_rate": 10 ** random.uniform(-4, -3),
            "margin": random.uniform(0.5, 1.2),
            "epochs": random.choice([5, 7, 10])
        }
        for _ in range(size)
    ]

def mutate(individual):
    mutated = copy.deepcopy(individual)
    if random.random() < 0.5:
        mutated["learning_rate"] *= random.uniform(0.8, 1.2)
    if random.random() < 0.3:
        mutated["margin"] += random.uniform(-0.1, 0.1)
    if random.random() < 0.3:
        mutated["batch_size"] = random.choice([16, 32])
    return mutated

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def select_parents(population, fitnesses, k=3):
    selected = random.choices(list(zip(population, fitnesses)), k=k)
    return min(selected, key=lambda x: x[1])[0]

def evolutionary_search(X1_train, X2_train, y_train, X1_val, X2_val, y_val, gens=1, pop_size=3):
    population = initialize_population(pop_size)
    best_model, best_base, best_loss, best_history, best_params = None, None, float('inf'), None, None

    for gen in range(gens):
        print(f"🧬 Generation {gen+1}/{gens}")
        fitnesses = []
        for i, individual in enumerate(population):
            loss, model, base, history = train_and_evaluate(
                individual, X1_train, X2_train, y_train, X1_val, X2_val, y_val)
            print(f"  Candidate {i+1}: Val Loss = {loss:.4f}")
            fitnesses.append(loss)
            if loss < best_loss:
                best_loss, best_model, best_base, best_history = loss, model, base, history
                best_params = individual

        new_population = []
        for _ in range(pop_size):
            p1 = select_parents(population, fitnesses)
            p2 = select_parents(population, fitnesses)
            child = mutate(crossover(p1, p2))
            new_population.append(child)

        population = new_population

    return best_model, best_base, best_history, best_params

# ----- Main -----
if __name__ == "__main__":
    os.makedirs("training_logs_ea", exist_ok=True)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = load_dataset(max_samples=1000)

    best_model, best_base_model, history, best_params = evolutionary_search(
        X1_train, X2_train, y_train, X1_val, X2_val, y_val,
        gens=1, pop_size=3
    )

    best_base_model.save("base_siamese_model_evolutionary.h5")

    # Save training plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Best Contrastive Loss over Epochs')
    plt.legend()
    plt.savefig("training_logs_ea/loss_curve.png")

    with open("training_logs_ea/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    np.save("training_logs_ea/history.npy", history.history)
    pd.DataFrame(history.history).to_csv("training_logs_ea/training_log.csv", index_label="epoch")

    print(f"\n✅ Best parameters found: {best_params}")
    print(f"📉 Final validation loss: {min(history.history['val_loss']):.4f}")
