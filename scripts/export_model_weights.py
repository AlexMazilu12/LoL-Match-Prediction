"""Export model weights for JavaScript client-side prediction."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Export Logistic Regression weights
print("Exporting Logistic Regression weights...")
lr_model = joblib.load(PROJECT_ROOT / "models" / "main_model.joblib")
prep = lr_model.named_steps["prep"]
clf = lr_model.named_steps["clf"]

# Get feature names after preprocessing
feature_names = prep.get_feature_names_out().tolist()
coefficients = clf.coef_[0].tolist()
intercept = float(clf.intercept_[0])

# Get scaler parameters for numeric features
numeric_features = ["gold_diff_15", "cs_diff_15"]
num_transformer = prep.named_transformers_["num"]
# The numeric transformer is a Pipeline - get the last step (StandardScaler)
scaler = num_transformer[-1]  # Get the last step in the pipeline
scaler_mean = scaler.mean_.tolist()
scaler_scale = scaler.scale_.tolist()

lr_export = {
    "model_name": "Logistic Regression (pruned)",
    "feature_names": feature_names,
    "coefficients": coefficients,
    "intercept": intercept,
    "numeric_features": numeric_features,
    "scaler_mean": scaler_mean,
    "scaler_scale": scaler_scale,
    "categorical_features": ["first_dragon", "first_tower", "first_herald"],
}

with open(PROJECT_ROOT / "models" / "logistic_weights.json", "w") as f:
    json.dump(lr_export, f, indent=2)

print(f"✓ Saved Logistic Regression weights with {len(feature_names)} features")

# Export model metadata
with open(PROJECT_ROOT / "models" / "main_model_meta.json", "r") as f:
    lr_meta = json.load(f)

# Create a simplified RF metadata (since we can't load the full model)
rf_meta = {
    "model_name": "Random Forest (full)",
    "feature_set": "full",
    "features": json.load(open(PROJECT_ROOT / "models" / "rf_final_features.json")),
    "metrics": {
        "accuracy": 0.7519,  # Approximate from notebook
        "precision": 0.75,
        "recall": 0.77,
        "f1": 0.76,
        "roc_auc": 0.83,
    },
}

# Export combined model config
models_config = {
    "models": {
        "logistic_pruned": {
            "name": "Logistic Regression (pruned)",
            "features": lr_export["numeric_features"] + lr_export["categorical_features"],
            "metrics": lr_meta["metrics"],
            "weights_file": "logistic_weights.json",
        },
        "random_forest_full": {
            "name": "Random Forest (full)",
            "features": rf_meta["features"],
            "metrics": rf_meta["metrics"],
            "note": "Client-side approximation using feature importances",
        },
    }
}

with open(PROJECT_ROOT / "models" / "models_config.json", "w") as f:
    json.dump(models_config, f, indent=2)

print("✓ Saved models configuration")
print("\nExport complete!")
