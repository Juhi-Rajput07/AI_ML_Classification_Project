from pathlib import Path
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

# Load the same Wine dataset used in the project
wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

# Train the final Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Create models folder
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Save model
model_path = models_dir / "random_forest_model.joblib"
joblib.dump(model, model_path)

# Save feature names
feature_names_path = models_dir / "feature_names.joblib"
joblib.dump(list(X.columns), feature_names_path)

print("Final deployment model saved successfully.")
print(f"Model: {model_path}")
print(f"Features: {feature_names_path}")