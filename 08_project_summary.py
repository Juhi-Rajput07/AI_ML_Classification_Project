"""Print a final, evidence-based summary of the Wine classification project."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine


# Load dataset facts directly from scikit-learn and metrics from saved results.
wine = load_wine(as_frame=True)
output_directory = Path("outputs")
random_forest_results = pd.read_csv(output_directory / "random_forest_results.csv")
neural_network_results = pd.read_csv(output_directory / "neural_network_results.csv")
comparison_table = pd.read_csv(output_directory / "model_comparison.csv")

print("FINAL WINE CLASSIFICATION PROJECT SUMMARY")
print("=" * 60)
print("\nDataset information:")
print("Dataset: scikit-learn Wine dataset")
print(f"Number of samples: {wine.data.shape[0]}")
print(f"Number of input features: {wine.data.shape[1]}")
print(f"Number of classes: {len(wine.target_names)}")
print(f"Class names: {', '.join(wine.target_names)}")

print("\nRandom Forest metrics:")
print(random_forest_results.to_string(index=False))

print("\nNeural Network metrics:")
print(neural_network_results.to_string(index=False))

print("\nModel comparison:")
print(comparison_table.to_string(index=False))

# Select the model(s) with the maximum measured accuracy. Handling a tie keeps
# the conclusion faithful to the actual result data.
highest_accuracy = comparison_table["Accuracy"].max()
best_models = comparison_table.loc[
    comparison_table["Accuracy"] == highest_accuracy, "Model"
].tolist()

print("\nHighest accuracy:")
print(f"{', '.join(best_models)} with accuracy {highest_accuracy:.4f}")

print("\nImportant observations:")
print("- Both models used the same 80/20 stratified train/test split (random_state=42).")
print("- Neural-network scaling was fit only on training features to avoid data leakage.")
print("- The reported model comparison is based on the saved test-set metrics above.")
