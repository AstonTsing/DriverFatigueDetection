from __future__ import annotations

import json
import sys
from pathlib import Path


def write_response(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import predictor

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "preload":
                write_response({"success": True, "cmd": "preload", "status": predictor.preload_all()})
            elif command == "predict":
                method = request["method"]
                image = Path(request["image"])
                write_response(predictor.predict_with_method(method, image))
            elif command == "compare":
                image = Path(request["image"])
                methods = request.get("methods") or predictor.get_available_methods()
                results = [predictor.predict_with_method(method, image) for method in methods]
                write_response(
                    {
                        "comparison": True,
                        "method": "五模型对比检测",
                        "image_path": str(image),
                        "success": any(result.get("success") for result in results),
                        "results": results,
                    }
                )
            elif command == "methods":
                write_response({"success": True, "methods": predictor.get_available_methods()})
            elif command == "exit":
                write_response({"success": True, "cmd": "exit"})
                return 0
            else:
                write_response({"success": False, "error": f"未知命令: {command}"})
        except Exception as exc:
            write_response({"success": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
