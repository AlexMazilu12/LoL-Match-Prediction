from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression win predictor.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split fraction.")
    parser.add_argument("--skip-preview", action="store_true", help="Do not score the preview sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from lol_prediction import config, data, modeling, validation

    dataset = data.load_15m_dataset()
    result = modeling.train_logistic(dataset, test_size=args.test_size)
    modeling.save_model(result.model)
    metadata = {
        "model_name": "Logistic Regression (pruned)",
        "feature_set": "pruned",
        "features": result.feature_names,
        "metrics": result.metrics,
    }
    data.save_metadata(metadata)
    print("Model trained and saved to", config.MODEL_PATH)
    print("Evaluation metrics:")
    print(json.dumps(result.metrics, indent=2))
    if not args.skip_preview:
        preview_metrics, _ = validation.validate_preview(model=result.model, save=True)
        print("Preview sample metrics (first", config.PREVIEW_LIMIT, "rows):")
        print(json.dumps(preview_metrics, indent=2))
        print("Preview predictions saved to", config.PREVIEW_PREDICTIONS)


if __name__ == "__main__":
    main()
