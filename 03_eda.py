"""Create the Wine dataset target-class distribution chart.

This EDA step visualizes only the number of samples in each target class.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine


# Load the built-in Wine dataset as pandas objects.
wine = load_wine(as_frame=True)

# Create one DataFrame containing the 13 features and the target column.
df = wine.data.copy()
df["target"] = wine.target

# Count the samples for target values 0, 1, and 2 in their natural order.
class_counts = df["target"].value_counts().sort_index()

# Match each numeric target value to the class name supplied by scikit-learn.
class_labels = list(wine.target_names)
chart_data = pd.DataFrame(
    {"Wine Class": class_labels, "Number of Samples": class_counts.values}
)

# Create the output folder automatically when it does not already exist.
output_directory = Path("outputs/figures")
output_directory.mkdir(parents=True, exist_ok=True)
output_file = output_directory / "01_class_distribution.png"

# Set a clean visual style and create the bar chart.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
bar_chart = sns.barplot(
    data=chart_data,
    x="Wine Class",
    y="Number of Samples",
    hue="Wine Class",
    palette="Blues_d",
    legend=False,
)

# Add the required title and axis labels.
bar_chart.set_title("Wine Dataset Class Distribution")
bar_chart.set_xlabel("Wine Class")
bar_chart.set_ylabel("Number of Samples")

# Print the exact count directly above every bar.
for bar in bar_chart.patches:
    count = int(bar.get_height())
    bar_chart.annotate(
        str(count),
        (bar.get_x() + bar.get_width() / 2, count),
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points",
    )

# Save the completed figure at high resolution and close it after saving.
# Closing the figure keeps the script suitable for non-interactive environments.
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print("Class counts:")
for class_name, count in zip(class_labels, class_counts.values):
    print(f"{class_name}: {count}")
print(f"\nFigure successfully saved to: {output_file}")


# Select only the 13 numerical input features. The target column is not
# included because it represents the class label rather than a measurement.
input_features = df[wine.feature_names]

# Create a 4 x 4 subplot grid, which provides room for all 13 features.
figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 16))
axes = axes.flatten()

# Draw a histogram and KDE curve for each feature.
for axis, feature_name in zip(axes, input_features.columns):
    sns.histplot(data=input_features, x=feature_name, kde=True, ax=axis)
    axis.set_title(feature_name)
    axis.set_xlabel(feature_name)
    axis.set_ylabel("Frequency")

# Remove the three unused subplot spaces from the 4 x 4 grid.
for axis in axes[len(input_features.columns) :]:
    figure.delaxes(axis)

# Add an overall title and arrange the subplots so labels do not overlap.
figure.suptitle("Distribution of Wine Dataset Features", fontsize=18)
figure.tight_layout(rect=(0, 0, 1, 0.97))

# Save the distribution figure in the same automatically created folder.
feature_distribution_file = output_directory / "02_feature_distributions.png"
figure.savefig(feature_distribution_file, dpi=300)
plt.close(figure)

print(f"Figure successfully saved to: {feature_distribution_file}")


# Calculate Pearson correlations using only the 13 input measurements.
# The target class column is deliberately excluded from this calculation.
correlation_matrix = input_features.corr(method="pearson")

print("\nPearson correlation matrix for the 13 input features:")
# Show every feature column when printing this wide matrix in the terminal.
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(correlation_matrix)

# Create a readable, annotated heatmap of the feature correlations.
plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation"},
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()

# Save the correlation heatmap in the existing output directory.
correlation_heatmap_file = output_directory / "03_correlation_heatmap.png"
plt.savefig(correlation_heatmap_file, dpi=300)
plt.close()

print(f"\nHeatmap successfully saved to: {correlation_heatmap_file}")


# Plot the relationship between flavanoids and total phenols. Point colours
# identify the Wine class without changing the original target values.
class_name_map = dict(enumerate(wine.target_names))
relationship_data = df.assign(wine_class=df["target"].map(class_name_map))

plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=relationship_data,
    x="flavanoids",
    y="total_phenols",
    hue="wine_class",
    palette="deep",
)
plt.title("Flavanoids vs Total Phenols by Wine Class")
plt.xlabel("Flavanoids")
plt.ylabel("Total Phenols")
plt.legend(title="Wine Class")
plt.tight_layout()

# Save this single feature-relationship plot in the existing figures folder.
feature_relationship_file = output_directory / "04_feature_relationship.png"
plt.savefig(feature_relationship_file, dpi=300)
plt.close()

print(f"Feature relationship figure successfully saved to: {feature_relationship_file}")
