"""Train and evaluate a Random Forest classifier on the Wine dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Load the original dataset and separate feature measurements from class labels.
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

# Use the same reproducible, stratified split planned for every model.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Train the requested Random Forest model.
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

# Calculate metrics from the actual test predictions.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("RANDOM FOREST RESULTS")
print("=" * 50)
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

# Plot the confusion matrix using readable class names.
matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=wine.target_names,
    yticklabels=wine.target_names,
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
confusion_matrix_file = figures_directory / "05_random_forest_confusion_matrix.png"
plt.savefig(confusion_matrix_file, dpi=300)
plt.close()

# Plot feature importances from the fitted Random Forest model.
importance_data = pd.DataFrame(
    {"Feature": X.columns, "Importance": random_forest.feature_importances_}
).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=importance_data,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False,
)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
importance_file = figures_directory / "06_random_forest_feature_importance.png"
plt.savefig(importance_file, dpi=300)
plt.close()

# Save the actual metrics so later scripts can compare the models.
results = pd.DataFrame(
    [{
        "Model": "Random Forest",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }]
)
results_file = output_directory / "random_forest_results.csv"
results.to_csv(results_file, index=False)

print(f"Confusion matrix saved to: {confusion_matrix_file}")
print(f"Feature importance plot saved to: {importance_file}")
print(f"Metrics saved to: {results_file}")
