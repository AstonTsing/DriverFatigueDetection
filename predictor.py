"""统一驾驶员疲劳检测预测接口，带模型缓存和便携环境兼容修复。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent
V0_DIR = BASE_DIR / "DriverFatigueDetection" / "DriverFatigueDetection-v0"
V1_DIR = BASE_DIR / "DriverFatigueDetection" / "DriverFatigueDetection-v1"
PYTHON_ENV = BASE_DIR / "python_env"
SITE_PACKAGES = PYTHON_ENV / "Lib" / "site-packages"
_CACHE: dict[str, Any] = {}


def _add_dll_directories() -> None:
    """Add portable dependency DLL folders for pythonw/GUI launches."""
    candidates = [
        PYTHON_ENV,
        SITE_PACKAGES,
        SITE_PACKAGES / "mediapipe" / "python",
        SITE_PACKAGES / "PyQt5" / "Qt5" / "bin",
        SITE_PACKAGES / "cv2",
        SITE_PACKAGES / "tensorflow",
        SITE_PACKAGES / "torch" / "lib",
        SITE_PACKAGES / "numpy.libs",
        SITE_PACKAGES / "scipy.libs",
        SITE_PACKAGES / "sklearn" / ".libs",
    ]
    existing = [str(path) for path in candidates if path.exists()]
    for path in existing:
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError):
            pass
    os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])


_add_dll_directories()


def _use_project_src(project_dir: Path) -> None:
    """Switch the ambiguous `src` package import to one project."""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    project_str = str(project_dir)
    sys.path[:] = [path for path in sys.path if path not in {str(V0_DIR), str(V1_DIR)}]
    sys.path.insert(0, project_str)


def _prepare_mediapipe_ascii_resources() -> None:
    """Make MediaPipe resources available from an ASCII-only path.

    MediaPipe's native runtime can fail when resource paths contain Chinese
    characters. Copying its resource folders to %TEMP% and pointing SolutionBase
    there avoids that issue while keeping the project portable.
    """
    if _CACHE.get("mediapipe_resources_ready"):
        return
    import mediapipe
    import mediapipe.python.solution_base as solution_base

    source_root = Path(mediapipe.__file__).resolve().parent
    target_root = Path(tempfile.gettempdir()) / "driver_fatigue_mediapipe" / "mediapipe"
    for name in ("modules", "tasks"):
        source = source_root / name
        target = target_root / name
        if source.exists() and not target.exists():
            shutil.copytree(source, target)
    python_dir = target_root / "python"
    python_dir.mkdir(parents=True, exist_ok=True)
    fake_solution = python_dir / "solution_base.py"
    if not fake_solution.exists():
        fake_solution.write_text("# fake path for mediapipe resource root\n", encoding="utf-8")
    solution_base.__file__ = str(fake_solution)
    _CACHE["mediapipe_resources_ready"] = True


def _base_result(method: str, image_path: str | Path) -> dict[str, Any]:
    return {"method": method, "image_path": str(image_path)}


def _load_v0() -> dict[str, Any]:
    if "v0" not in _CACHE:
        _use_project_src(V0_DIR)
        _prepare_mediapipe_ascii_resources()
        import joblib
        from src.dataset import LABEL_TO_CLASS
        from src.landmark_features import FaceLandmarkFeatureExtractor, read_image
        from src.rule_detector import RuleConfig

        config = RuleConfig.from_json(V0_DIR / "config" / "rule_config.json")
        extractor = FaceLandmarkFeatureExtractor(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        _CACHE["v0"] = {
            "LABEL_TO_CLASS": LABEL_TO_CLASS,
            "read_image": read_image,
            "extractor": extractor,
            "model": joblib.load(V0_DIR / "models" / "svm_ear_mar.joblib"),
        }
    return _CACHE["v0"]


def predict_v0_svm(image_path: str | Path) -> dict[str, Any]:
    method = "DriverFatigueDetection-v0 (SVM)"
    try:
        data = _load_v0()
        image = data["read_image"](str(image_path))
        features = data["extractor"].extract(image)
        result = _base_result(method, image_path)
        if not features.face_found:
            result.update({"success": False, "error": "未检测到人脸"})
            return result
        x = [[features.ear_left, features.ear_right, features.ear_mean, features.mar]]
        model = data["model"]
        pred = int(model.predict(x)[0])
        proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
        label_map = data["LABEL_TO_CLASS"]
        result.update(
            {
                "success": True,
                "prediction": {
                    "label": label_map[pred],
                    "is_drowsy": bool(pred),
                    "confidence": float(proba[pred]) if proba is not None else None,
                    "probabilities": {
                        "notdrowsy": float(proba[0]) if proba is not None else None,
                        "drowsy": float(proba[1]) if proba is not None else None,
                    },
                },
                "features": {
                    "ear_left": float(features.ear_left),
                    "ear_right": float(features.ear_right),
                    "ear_mean": float(features.ear_mean),
                    "mar": float(features.mar),
                },
            }
        )
        return result
    except Exception as exc:
        result = _base_result(method, image_path)
        result.update({"success": False, "error": str(exc)})
        return result


def _load_v1() -> dict[str, Any]:
    if "v1" not in _CACHE:
        _use_project_src(V1_DIR)
        _prepare_mediapipe_ascii_resources()
        import joblib
        from src.dataset import LABEL_TO_CLASS
        from src.feature_extractor import RichFeatureExtractor
        from src.image_io import read_image
        from src.metrics import predict_prob_drowsy

        with open(V1_DIR / "config" / "model_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
        extractor = RichFeatureExtractor(
            min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )
        bundle = joblib.load(V1_DIR / "models" / "hgb_fatigue.joblib")
        _CACHE["v1"] = {
            "LABEL_TO_CLASS": LABEL_TO_CLASS,
            "read_image": read_image,
            "extractor": extractor,
            "predict_prob_drowsy": predict_prob_drowsy,
            "model": bundle["pipeline"],
        }
    return _CACHE["v1"]


def predict_v1_hgb(image_path: str | Path) -> dict[str, Any]:
    method = "DriverFatigueDetection-v1 (HistGradientBoosting)"
    try:
        data = _load_v1()
        image = data["read_image"](str(image_path))
        extracted = data["extractor"].extract(image)
        result = _base_result(method, image_path)
        if not extracted.face_found:
            result.update({"success": False, "error": "未检测到人脸"})
            return result
        x = [extracted.feature_vector()]
        model = data["model"]
        pred = int(model.predict(x)[0])
        prob_drowsy = data["predict_prob_drowsy"](model, x)
        proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
        label_map = data["LABEL_TO_CLASS"]
        result.update(
            {
                "success": True,
                "prediction": {
                    "label": label_map[pred],
                    "is_drowsy": bool(pred),
                    "prob_drowsy": float(prob_drowsy[0]) if prob_drowsy is not None else None,
                    "confidence": float(proba[pred]) if proba is not None else None,
                    "probabilities": {
                        "notdrowsy": float(proba[0]) if proba is not None else None,
                        "drowsy": float(proba[1]) if proba is not None else None,
                    },
                },
                "features": extracted.features,
            }
        )
        return result
    except Exception as exc:
        result = _base_result(method, image_path)
        result.update({"success": False, "error": str(exc)})
        return result


def _select_torch_device() -> str | None:
    try:
        import torch

        return "0" if torch.cuda.is_available() else None
    except Exception:
        return None


def _load_yolo():
    if "yolo" not in _CACHE:
        from ultralytics import YOLO

        _CACHE["yolo_device"] = _select_torch_device()
        _CACHE["yolo"] = YOLO(str(BASE_DIR / "drowsiness_detection" / "outputs" / "test1" / "best.pt"))
    return _CACHE["yolo"], _CACHE.get("yolo_device")


class RealtimeFaceDetector:
    """Face-box detector for webcam frames, with OpenCV fallback."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self.detector = None
        self.mesh = None
        self.haar = None
        self.backend = "none"
        self.errors: list[str] = []
        try:
            _prepare_mediapipe_ascii_resources()
            import mediapipe as mp

            try:
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )
                self.backend = "mediapipe-face-detection"
            except Exception as exc:
                self.errors.append(f"mediapipe-face-detection: {exc}")
                self.detector = None
                try:
                    self.mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=0.5,
                    )
                    self.backend = "mediapipe-face-mesh"
                except Exception as mesh_exc:
                    self.errors.append(f"mediapipe-face-mesh: {mesh_exc}")
                    self.mesh = None
        except Exception as exc:
            self.errors.append(f"mediapipe-import: {exc}")

        if self.backend == "none":
            self._load_haar_fallback()

    def _load_haar_fallback(self) -> None:
        try:
            import cv2

            cascade_names = [
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_frontalface_alt.xml",
                "haarcascade_frontalface_alt_tree.xml",
            ]
            source_dir = Path(cv2.data.haarcascades)
            temp_dir = Path(tempfile.gettempdir()) / "driver_fatigue_cv2_haar"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for name in cascade_names:
                source = source_dir / name
                if not source.exists():
                    self.errors.append(f"opencv-haar: missing cascade at {source}")
                    continue
                ascii_path = temp_dir / name
                try:
                    if not ascii_path.exists() or ascii_path.stat().st_size != source.stat().st_size:
                        shutil.copy2(source, ascii_path)
                except Exception as copy_exc:
                    self.errors.append(f"opencv-haar-copy {name}: {copy_exc}")
                    ascii_path = source
                for candidate in (ascii_path, source):
                    haar = cv2.CascadeClassifier(str(candidate))
                    if not haar.empty():
                        self.haar = haar
                        self.backend = f"opencv-haar:{name}"
                        return
                self.errors.append(f"opencv-haar: empty cascade {name}")
        except Exception as exc:
            self.haar = None
            self.errors.append(f"opencv-haar: {exc}")

    def detect(self, frame_bgr) -> tuple[int, int, int, int] | None:
        height, width = frame_bgr.shape[:2]
        if self.detector is not None:
            import cv2

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.detector.process(frame_rgb)
            detections = result.detections or []
            if detections:
                detection = max(
                    detections,
                    key=lambda item: item.score[0] if item.score else 0.0,
                )
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * width)
                y = int(box.ymin * height)
                w = int(box.width * width)
                h = int(box.height * height)
                return _clamp_box((x, y, w, h), width, height)
        if self.mesh is not None:
            import cv2

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.mesh.process(frame_rgb)
            faces = result.multi_face_landmarks or []
            if faces:
                landmarks = faces[0].landmark
                xs = [point.x for point in landmarks]
                ys = [point.y for point in landmarks]
                x1 = int(min(xs) * width)
                y1 = int(min(ys) * height)
                x2 = int(max(xs) * width)
                y2 = int(max(ys) * height)
                return _clamp_box((x1, y1, x2 - x1, y2 - y1), width, height)
        if self.haar is not None:
            import cv2

            try:
                if self.haar.empty():
                    self.haar = None
                    self.backend = "none"
                    return None
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            except cv2.error:
                self.haar = None
                self.backend = "none"
                return None
            if len(faces):
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                return _clamp_box((int(x), int(y), int(w), int(h)), width, height)
        return None

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _expand_box(box: tuple[int, int, int, int], width: int, height: int, margin_ratio: float = 0.25) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    return _clamp_box((x - margin_x, y - margin_y, w + margin_x * 2, h + margin_y * 2), width, height)


def predict_yolo_frame(frame_bgr, face_box: tuple[int, int, int, int] | None) -> dict[str, Any]:
    method = "YOLOv11 Classification"
    result = {"method": method, "image_path": "<webcam>", "face_found": face_box is not None}
    if face_box is None:
        result.update({"success": False, "error": "未检测到人脸"})
        return result
    try:
        model, device = _load_yolo()
        height, width = frame_bgr.shape[:2]
        crop_box = _expand_box(face_box, width, height) or _clamp_box(face_box, width, height)
        if crop_box is None:
            result.update({"success": False, "error": "人脸框无效"})
            return result
        x, y, w, h = crop_box
        crop = frame_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            result.update({"success": False, "error": "人脸区域为空"})
            return result
        kwargs = {"imgsz": 224, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        start = time.perf_counter()
        prediction = model.predict(crop, **kwargs)[0]
        inference_ms = (time.perf_counter() - start) * 1000.0
        pred_idx = int(prediction.probs.top1)
        pred_name = prediction.names[pred_idx]
        probs = prediction.probs.data.cpu().numpy()
        result.update(
            {
                "success": True,
                "device": "cuda:0" if device is not None else "cpu",
                "face_found": True,
                "face_box": {"x": int(face_box[0]), "y": int(face_box[1]), "w": int(face_box[2]), "h": int(face_box[3])},
                "crop_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "timing": {"inference_ms": float(inference_ms)},
                "prediction": {
                    "label": pred_name,
                    "is_drowsy": pred_name == "drowsy",
                    "confidence": float(prediction.probs.top1conf),
                    "probabilities": {prediction.names[i]: float(probs[i]) for i in range(len(probs))},
                },
            }
        )
        return result
    except Exception as exc:
        result.update({"success": False, "error": str(exc)})
        return result


def predict_yolo(image_path: str | Path) -> dict[str, Any]:
    method = "YOLOv11 Classification"
    try:
        model, device = _load_yolo()
        kwargs = {"imgsz": 224, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        prediction = model.predict(str(image_path), **kwargs)[0]
        pred_idx = int(prediction.probs.top1)
        pred_name = prediction.names[pred_idx]
        probs = prediction.probs.data.cpu().numpy()
        result = _base_result(method, image_path)
        result.update(
            {
                "success": True,
                "device": "cuda:0" if device is not None else "cpu",
                "prediction": {
                    "label": pred_name,
                    "is_drowsy": pred_name == "drowsy",
                    "confidence": float(prediction.probs.top1conf),
                    "probabilities": {prediction.names[i]: float(probs[i]) for i in range(len(probs))},
                },
            }
        )
        return result
    except Exception as exc:
        result = _base_result(method, image_path)
        result.update({"success": False, "error": str(exc)})
        return result


def _load_tf_model(model_file: str):
    cache_key = f"tf:{model_file}"
    if cache_key not in _CACHE:
        import tensorflow as tf

        model_path = BASE_DIR / "Introduction_to_AI-Project_chenbh" / model_file

        class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("groups", None)
                return super().from_config(config)

        if model_file == "model_mobilenetv2.h5":
            mobilenet_base = tf.keras.applications.MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3),
            )
            mobilenet_base.trainable = False
            model = tf.keras.Sequential(
                [
                    mobilenet_base,
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                    tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                    tf.keras.layers.Dense(2, activation="softmax"),
                ]
            )
            model.load_weights(str(model_path))
        else:
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
                compile=False,
            )
        _CACHE[cache_key] = model
    return _CACHE[cache_key]


def _predict_tf(image_path: str | Path, model_file: str, method: str) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np

        model = _load_tf_model(model_file)
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("无法读取图片")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, 0)
        probs = model.predict(image, verbose=0)[0]
        labels = ["drowsy", "notdrowsy"]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        result = _base_result(method, image_path)
        result.update(
            {
                "success": True,
                "prediction": {
                    "label": pred_label,
                    "is_drowsy": pred_label == "drowsy",
                    "confidence": float(probs[pred_idx]),
                    "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
                },
            }
        )
        return result
    except Exception as exc:
        result = _base_result(method, image_path)
        result.update({"success": False, "error": str(exc)})
        return result


def predict_tf_baseline(image_path: str | Path) -> dict[str, Any]:
    return _predict_tf(image_path, "model_baseline.h5", "TensorFlow Baseline CNN")


def predict_tf_mobilenet(image_path: str | Path) -> dict[str, Any]:
    return _predict_tf(image_path, "model_mobilenetv2.h5", "TensorFlow MobileNetV2")


PREDICTORS: dict[str, tuple[Callable[[str | Path], dict[str, Any]], Path]] = {
    "DriverFatigueDetection-v0 (SVM)": (predict_v0_svm, V0_DIR / "models" / "svm_ear_mar.joblib"),
    "DriverFatigueDetection-v1 (HistGradientBoosting)": (predict_v1_hgb, V1_DIR / "models" / "hgb_fatigue.joblib"),
    "YOLOv11 Classification": (predict_yolo, BASE_DIR / "drowsiness_detection" / "outputs" / "test1" / "best.pt"),
    "TensorFlow Baseline CNN": (predict_tf_baseline, BASE_DIR / "Introduction_to_AI-Project_chenbh" / "model_baseline.h5"),
    "TensorFlow MobileNetV2": (predict_tf_mobilenet, BASE_DIR / "Introduction_to_AI-Project_chenbh" / "model_mobilenetv2.h5"),
}


def get_available_methods() -> list[str]:
    return [name for name, (_, model_path) in PREDICTORS.items() if model_path.exists()]


def predict_with_method(method_name: str, image_path: str | Path) -> dict[str, Any]:
    if method_name not in PREDICTORS:
        return {"method": method_name, "image_path": str(image_path), "success": False, "error": "未知方法"}
    predictor, _ = PREDICTORS[method_name]
    return predictor(image_path)


def preload_all() -> list[dict[str, Any]]:
    """Preload all available model objects into memory for faster prediction."""
    status: list[dict[str, Any]] = []
    loaders: dict[str, Callable[[], Any]] = {
        "DriverFatigueDetection-v0 (SVM)": _load_v0,
        "DriverFatigueDetection-v1 (HistGradientBoosting)": _load_v1,
        "YOLOv11 Classification": _load_yolo,
        "TensorFlow Baseline CNN": lambda: _load_tf_model("model_baseline.h5"),
        "TensorFlow MobileNetV2": lambda: _load_tf_model("model_mobilenetv2.h5"),
    }
    for method in get_available_methods():
        try:
            loaders[method]()
            status.append({"method": method, "success": True})
        except Exception as exc:
            status.append({"method": method, "success": False, "error": str(exc)})
    return status


if __name__ == "__main__":
    print("可用方法:")
    for method in get_available_methods():
        print("-", method)
