from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run one fatigue detection prediction and print JSON.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    import predictor

    result = predictor.predict_with_method(args.method, Path(args.image))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
