"""Set up and inspect the scikit-learn Wine dataset.

This is the first step of a multiclass-classification project.  It loads the
dataset and performs basic data-quality checks; no model or visualisation is
created here.
"""

import pandas as pd
from sklearn.datasets import load_wine


# Load the built-in Wine dataset as pandas objects.
wine = load_wine(as_frame=True)

# Create a DataFrame with the 13 input feature columns.
df = wine.data.copy()

# Add the target column, which identifies the wine class for each row.
df["target"] = wine.target


# Display the requested basic dataset information.
print("WINE DATASET SETUP")
print("=" * 50)
print(f"\nDataset shape (rows, columns): {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nTarget class names:")
print(list(wine.target_names))

print("\nTarget value counts:")
print(df["target"].value_counts().sort_index())


# Detailed inspection of the input features. The target is excluded here because
# it is a class label, not a numerical measurement of the wine.
feature_names = wine.feature_names
numerical_features = df[feature_names]

print("\nDETAILED DATASET INSPECTION")
print("=" * 50)

print("\nAll feature names:")
for number, feature_name in enumerate(feature_names, start=1):
    print(f"{number}. {feature_name}")

print("\nTarget column name:")
print("target")

print("\nData types of all columns:")
print(df.dtypes)

print("\nDescriptive statistics for numerical features:")
# Show every feature column instead of shortening a wide table with "...".
with pd.option_context("display.max_columns", None):
    print(numerical_features.describe())

# Print one common statistic at a time to make the values easier to inspect.
print("\nMinimum value of every numerical feature:")
print(numerical_features.min())

print("\nMaximum value of every numerical feature:")
print(numerical_features.max())

print("\nMean of every numerical feature:")
print(numerical_features.mean())

print("\nMedian of every numerical feature:")
print(numerical_features.median())

print("\nStandard deviation of every numerical feature:")
print(numerical_features.std())

print("\nTarget class names and numeric target mapping:")
for target_value, class_name in enumerate(wine.target_names):
    print(f"{target_value} corresponds to {class_name}")

# Check data quality before using the dataset in later project steps.
print("\nMissing values per column:")
print(df.isnull().sum())

duplicate_rows = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_rows}")


# Print a concise summary of what was loaded.
print("\nDATASET SUMMARY")
print("-" * 50)
print(f"Total records: {df.shape[0]}")
print("Input features: 13")
print("Target column: target")
print(f"Number of target classes: {len(wine.target_names)}")
print(f"Missing values in dataset: {df.isnull().sum().sum()}")
print(f"Duplicate rows in dataset: {duplicate_rows}")
print("The Wine dataset is loaded and ready for the next project step.")
