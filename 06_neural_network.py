"""Train and evaluate a basic TensorFlow/Keras neural network on Wine data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Set practical random seeds so repeated runs are more reproducible.
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# Load and split the same Wine dataset used throughout the project.
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Fit the scaler only on training features, then transform both sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the requested feed-forward neural-network architecture.
model = Sequential(
    [
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Stop training when validation loss has not improved for 10 epochs, retaining
# the weights from the best validation-loss epoch.
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.20,
    callbacks=[early_stopping],
    verbose=0,
)

# Evaluate the trained model and calculate metrics from test-set predictions.
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
predicted_probabilities = model.predict(X_test_scaled, verbose=0)
y_pred = predicted_probabilities.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("NEURAL NETWORK RESULTS")
print("=" * 50)
print(f"Epochs completed: {len(history.history['loss'])}")
print(f"Test loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted precision: {precision:.4f}")
print(f"Weighted recall: {recall:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names, zero_division=0))

# Create output folders if they do not already exist.
output_directory = Path("outputs")
figures_directory = output_directory / "figures"
figures_directory.mkdir(parents=True, exist_ok=True)

# Save the neural-network confusion matrix.
matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=wine.target_names,
    yticklabels=wine.target_names,
)
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
confusion_matrix_file = figures_directory / "07_neural_network_confusion_matrix.png"
plt.savefig(confusion_matrix_file, dpi=300)
plt.close()

# Plot the actual training and validation accuracy history.
epochs = range(1, len(history.history["accuracy"]) + 1)
plt.figure(figsize=(9, 6))
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
accuracy_file = figures_directory / "08_training_validation_accuracy.png"
plt.savefig(accuracy_file, dpi=300)
plt.close()

# Plot the actual training and validation loss history.
plt.figure(figsize=(9, 6))
plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
loss_file = figures_directory / "09_training_validation_loss.png"
plt.savefig(loss_file, dpi=300)
plt.close()

# Save actual model metrics for the comparison and summary scripts.
results = pd.DataFrame(
    [{
        "Model": "Neural Network",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }]
)
results_file = output_directory / "neural_network_results.csv"
results.to_csv(results_file, index=False)

print(f"Confusion matrix saved to: {confusion_matrix_file}")
print(f"Accuracy history saved to: {accuracy_file}")
print(f"Loss history saved to: {loss_file}")
print(f"Metrics saved to: {results_file}")
