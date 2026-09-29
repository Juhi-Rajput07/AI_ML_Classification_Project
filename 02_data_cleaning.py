"""Check the quality of the scikit-learn Wine dataset.

This step only inspects data quality. It does not remove, replace, scale, or
otherwise modify any dataset values.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


# Load the built-in Wine dataset as pandas objects.
wine = load_wine(as_frame=True)

# Keep the 13 input features separate from the class-label column.
feature_names = wine.feature_names
df = wine.data.copy()
df["target"] = wine.target
numerical_features = df[feature_names]


print("WINE DATASET: DATA QUALITY AND CLEANING CHECKS")
print("=" * 60)

# Check every column for missing values.
missing_values = df.isnull().sum()
print("\nMissing values in every column:")
print(missing_values)

# Count rows that are exact duplicates of an earlier row.
duplicate_rows = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_rows}")

# Infinite values are not valid numeric measurements. Check the 13 features.
infinite_values = pd.Series(
    np.isinf(numerical_features).sum(), index=feature_names
)
print("\nInfinite values in every input feature:")
print(infinite_values)
print(f"Total infinite values: {infinite_values.sum()}")

# Show the observed minimum and maximum for each input feature.
feature_ranges = pd.DataFrame(
    {"minimum": numerical_features.min(), "maximum": numerical_features.max()}
)
print("\nNumerical range of every input feature:")
print(feature_ranges)


# Use the IQR method to flag possible outliers. Flagging does not mean a value
# is incorrect, so this script deliberately leaves all observations unchanged.
q1 = numerical_features.quantile(0.25)
q3 = numerical_features.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# A potential outlier falls below the lower bound or above the upper bound.
outlier_counts = (
    (numerical_features.lt(lower_bound)) | (numerical_features.gt(upper_bound))
).sum()

iqr_results = pd.DataFrame(
    {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "potential_outlier_count": outlier_counts,
    }
)

print("\nIQR outlier analysis for the 13 input features:")
# Display every IQR calculation column instead of shortening the wide table.
with pd.option_context("display.max_columns", None):
    print(iqr_results)

features_with_outliers = outlier_counts[outlier_counts > 0]


# Provide a clear conclusion without performing any automatic cleaning.
print("\nDATA-QUALITY SUMMARY")
print("-" * 60)
print(f"Missing values: {missing_values.sum()}")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Infinite values: {infinite_values.sum()}")

if features_with_outliers.empty:
    print("Features containing potential IQR outliers: None")
else:
    print("Features containing potential IQR outliers:")
    for feature_name, count in features_with_outliers.items():
        print(f"- {feature_name}: {count}")

print("Cleaning action required: No automatic action was taken.")
print("Potential outliers were reported only; all valid observations were kept.")
