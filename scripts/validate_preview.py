from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lol_prediction import config, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the preview dataset with the saved model.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing predictions to disk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, scored = validation.validate_preview(save=not args.no_save)
    print("Preview metrics:")
    print(json.dumps(metrics, indent=2))
    if not args.no_save:
        print("Preview predictions saved to", config.PREVIEW_PREDICTIONS)
    print("Rows scored:", len(scored))


if __name__ == "__main__":
    main()
