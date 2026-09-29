"""Split and scale the Wine dataset without data leakage.

The scaler is fit only on training features, ready for use by a neural network.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the original Wine dataset and separate the inputs from the target labels.
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

# Split before scaling. Stratification keeps the class proportions similar in
# the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("WINE DATASET PREPROCESSING")
print("=" * 50)
print(f"Training feature shape: {X_train.shape}")
print(f"Testing feature shape: {X_test.shape}")

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

# Fit the scaler on training data only. This prevents test-data information
# from leaking into the training process.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandardScaler fitted on X_train only.")
print(f"Scaled training feature shape: {X_train_scaled.shape}")
print(f"Scaled testing feature shape: {X_test_scaled.shape}")
