"""驾驶员疲劳检测系统图形界面。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_ENV = BASE_DIR / "python_env"
SITE_PACKAGES = PYTHON_ENV / "Lib" / "site-packages"
QT_PLUGIN_DIR = SITE_PACKAGES / "PyQt5" / "Qt5" / "plugins" / "platforms"
if QT_PLUGIN_DIR.exists():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(QT_PLUGIN_DIR))
    os.environ.setdefault("QT_PLUGIN_PATH", str(QT_PLUGIN_DIR.parent))
DLL_DIRS = [
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
EXISTING_DLL_DIRS = [str(path) for path in DLL_DIRS if path.exists()]
os.environ["PATH"] = os.pathsep.join(EXISTING_DLL_DIRS + [os.environ.get("PATH", "")])
for dll_dir in EXISTING_DLL_DIRS:
    try:
        os.add_dll_directory(dll_dir)
    except (AttributeError, OSError):
        pass

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from predictor import get_available_methods


class PredictionServerClient:
    """Persistent prediction process used by the GUI."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        python_exe = BASE_DIR / "python_env" / "python.exe"
        server_script = BASE_DIR / "prediction_server.py"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.process = subprocess.Popen(
            [str(python_exe), str(server_script)],
            cwd=str(BASE_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def request(self, payload: dict) -> dict:
        self.start()
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("预测服务未启动")
        if self.process.poll() is not None:
            raise RuntimeError(f"预测服务已退出，退出码 {self.process.returncode}")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("预测服务没有返回结果")
        return json.loads(line)

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.request({"cmd": "exit"})
                except Exception:
                    pass
                self.process.terminate()
        finally:
            self.process = None


class PreloadThread(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, server: PredictionServerClient):
        super().__init__()
        self.server = server

    def run(self) -> None:
        try:
            self.finished.emit(self.server.request({"cmd": "preload"}))
        except Exception as exc:
            self.finished.emit({"success": False, "error": str(exc)})


class PredictionThread(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, server: PredictionServerClient, methods: list[str], image_path: str, comparison: bool = False):
        super().__init__()
        self.server = server
        self.methods = methods
        self.image_path = image_path
        self.comparison = comparison

    def run(self) -> None:
        try:
            if self.comparison:
                self.finished.emit(
                    self.server.request({"cmd": "compare", "methods": self.methods, "image": self.image_path})
                )
            else:
                self.finished.emit(
                    self.server.request({"cmd": "predict", "method": self.methods[0], "image": self.image_path})
                )
        except Exception as exc:
            self.finished.emit({"method": "预测服务", "image_path": self.image_path, "success": False, "error": str(exc)})


class WebcamThread(QThread):
    frame_ready = pyqtSignal(object)
    result_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, server: PredictionServerClient, camera_index: int, target_fps: int):
        super().__init__()
        self.server = server
        self.camera_index = camera_index
        self.target_fps = max(1, target_fps)
        self.running = True
        self.last_result: dict | None = None

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        import cv2
        import predictor

        capture = None
        detector = None
        temp_dir = None
        frame_count = 0
        fps_started = time.perf_counter()
        interval = 1.0 / self.target_fps
        try:
            self.status_changed.emit("正在打开摄像头...")
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                self.result_ready.emit({"success": False, "error": f"无法打开摄像头 {self.camera_index}", "webcam": True})
                return
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.status_changed.emit("正在初始化人脸检测器并复用已预加载 YOLOv11 服务...")
            detector = predictor.RealtimeFaceDetector()
            temp_dir = tempfile.TemporaryDirectory(prefix="driver_fatigue_webcam_")
            temp_image = Path(temp_dir.name) / "webcam_face.jpg"
            self.status_changed.emit(f"摄像头检测运行中，人脸检测后端：{detector.backend}")
            while self.running:
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.result_ready.emit({"success": False, "error": "读取摄像头画面失败", "webcam": True})
                    break
                try:
                    face_box = detector.detect(frame)
                except Exception as exc:
                    face_box = None
                    detector.errors.append(f"detect: {exc}")
                estimated_face_box = False
                if face_box is None and detector.backend == "none":
                    face_box = self.estimate_face_box(frame)
                    estimated_face_box = True
                if face_box is None:
                    result = {"success": False, "error": "未检测到人脸", "webcam": True, "face_found": False}
                else:
                    crop = self.crop_face(frame, face_box)
                    if crop is None:
                        result = {"success": False, "error": "人脸区域为空", "webcam": True, "face_found": True}
                    else:
                        cv2.imencode(".jpg", crop)[1].tofile(str(temp_image))
                        infer_started = time.perf_counter()
                        result = self.server.request(
                            {"cmd": "predict", "method": "YOLOv11 Classification", "image": str(temp_image)}
                        )
                        result.setdefault("timing", {})["inference_ms"] = (time.perf_counter() - infer_started) * 1000.0
                        result["face_found"] = not estimated_face_box
                        result["estimated_face_box"] = estimated_face_box
                        result["face_box"] = {
                            "x": int(face_box[0]),
                            "y": int(face_box[1]),
                            "w": int(face_box[2]),
                            "h": int(face_box[3]),
                        }
                result.update(
                    {
                        "webcam": True,
                        "image_path": "<webcam>",
                        "target_fps": self.target_fps,
                        "face_detector": detector.backend,
                        "face_detector_errors": detector.errors[-3:],
                    }
                )
                frame_count += 1
                elapsed_for_fps = max(time.perf_counter() - fps_started, 1e-6)
                result["actual_fps"] = frame_count / elapsed_for_fps
                annotated = self.annotate_frame(frame, result)
                self.last_result = result
                self.frame_ready.emit(annotated)
                self.result_ready.emit(result)
                elapsed = time.perf_counter() - started
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))
        except Exception as exc:
            self.result_ready.emit({"success": False, "error": str(exc), "webcam": True})
        finally:
            if detector is not None:
                detector.close()
            if capture is not None:
                capture.release()
            if temp_dir is not None:
                temp_dir.cleanup()
            self.stopped.emit()

    def estimate_face_box(self, frame) -> tuple[int, int, int, int]:
        """Fallback box when all face detectors fail to initialize.

        This keeps real-time YOLO classification usable even if MediaPipe/OpenCV
        detector resources fail in the GUI launch environment.
        """
        height, width = frame.shape[:2]
        box_w = int(width * 0.42)
        box_h = int(height * 0.55)
        x = int((width - box_w) / 2)
        y = int(height * 0.25)
        return x, y, box_w, box_h

    def crop_face(self, frame, face_box: tuple[int, int, int, int]):
        height, width = frame.shape[:2]
        x, y, w, h = face_box
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.25)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + w + margin_x)
        y2 = min(height, y + h + margin_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def annotate_frame(self, frame, result: dict):
        import cv2

        annotated = frame.copy()
        if result.get("face_box"):
            box = result["face_box"]
            prediction = result.get("prediction") or {}
            success = bool(result.get("success"))
            is_drowsy = bool(prediction.get("is_drowsy"))
            if success:
                color = (0, 0, 255) if is_drowsy else (0, 180, 0)
                confidence = prediction.get("confidence")
                score = f" {confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else ""
                prefix = "estimated " if result.get("estimated_face_box") else ""
                label = f"{prefix}{prediction.get('label', 'unknown')}{score}"
            else:
                color = (0, 180, 255)
                label = "face detected / predicting"
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label_y = y - 10 if y >= 25 else y + h + 25
            cv2.putText(annotated, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2, cv2.LINE_AA)
        return annotated


class FatigueTimelineWidget(QWidget):
    """Rolling time-axis bar: red=drowsy, green=not drowsy, gray=unknown."""

    def __init__(self, window_seconds: float = 120.0):
        super().__init__()
        self.window_seconds = window_seconds
        self.samples: list[tuple[float, bool | None]] = []
        self.setMinimumHeight(34)
        self.setToolTip("最近一段时间的检测分布：红色=疲劳，绿色=清醒，灰色=未检测")

    def reset(self) -> None:
        self.samples.clear()
        self.update()

    def add_sample(self, is_drowsy: bool | None) -> None:
        now = time.perf_counter()
        self.samples.append((now, is_drowsy))
        cutoff = now - self.window_seconds
        self.samples = [(stamp, value) for stamp, value in self.samples if stamp >= cutoff]
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f5f5f5"))
        width = max(1, self.width())
        height = max(1, self.height())
        now = time.perf_counter()
        cutoff = now - self.window_seconds
        painter.setPen(QColor("#cccccc"))
        painter.drawRect(0, 0, width - 1, height - 1)
        if not self.samples:
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignCenter, "疲劳/清醒时间轴")
            return
        for stamp, value in self.samples:
            if stamp < cutoff:
                continue
            x = int(((stamp - cutoff) / self.window_seconds) * width)
            if value is True:
                color = QColor("#c62828")
            elif value is False:
                color = QColor("#2e7d32")
            else:
                color = QColor("#9e9e9e")
            painter.fillRect(max(0, x - 2), 2, 4, height - 4, color)
        painter.setPen(QColor("#555555"))
        painter.drawText(6, height - 8, f"最近 {int(self.window_seconds)} 秒")


class DetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path: str | None = None
        self.worker: PredictionThread | None = None
        self.preload_worker: PreloadThread | None = None
        self.webcam_worker: WebcamThread | None = None
        self.webcam_running = False
        self.last_webcam_text_update = 0.0
        self.server = PredictionServerClient()
        self.preload_done = False
        self.init_ui()
        self.on_mode_changed(self.mode_combo.currentText())
        self.start_preload()

    def init_ui(self) -> None:
        self.setWindowTitle("驾驶员疲劳检测系统")
        self.resize(1200, 800)

        root = QWidget()
        layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        layout.addWidget(self.build_control_panel(), 1)
        layout.addWidget(self.build_result_panel(), 2)

    def build_control_panel(self) -> QGroupBox:
        group = QGroupBox("控制面板")
        layout = QVBoxLayout(group)

        title = QLabel("驾驶员疲劳检测系统")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.preload_label = QLabel("模型预加载：准备启动...")
        self.preload_label.setWordWrap(True)
        self.preload_label.setStyleSheet("color: #c47f00;")
        layout.addWidget(self.preload_label)

        layout.addWidget(QLabel("选择检测功能："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["单模型检测", "五模型对比检测", "实时摄像头检测"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)

        layout.addWidget(QLabel("选择检测方法："))
        self.method_combo = QComboBox()
        self.available_methods = get_available_methods()
        if not self.available_methods:
            QMessageBox.critical(self, "错误", "未找到可用模型，请检查模型文件。")
        self.method_combo.addItems(self.available_methods)
        layout.addWidget(self.method_combo)

        self.camera_label = QLabel("选择摄像头：")
        layout.addWidget(self.camera_label)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2"])
        layout.addWidget(self.camera_combo)

        self.fps_label = QLabel("实时检测帧率：")
        layout.addWidget(self.fps_label)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["1", "3", "5", "10", "15", "30"])
        self.fps_combo.setCurrentText("30")
        layout.addWidget(self.fps_combo)

        layout.addWidget(QLabel("\n数据集默认位置："))
        dataset_label = QLabel("DriverFatigueDetection/dataset_split/test")
        dataset_label.setWordWrap(True)
        dataset_label.setStyleSheet("color: #555;")
        layout.addWidget(dataset_label)

        self.select_button = QPushButton("选择图片")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        self.path_label = QLabel("未选择图片")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #0057b8;")
        layout.addWidget(self.path_label)

        self.detect_button = QPushButton("开始单模型检测")
        self.detect_button.clicked.connect(self.start_detection)
        self.detect_button.setEnabled(False)
        self.detect_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; padding: 10px; font-size: 14pt; }"
            "QPushButton:disabled { background-color: #999; }"
        )
        layout.addWidget(self.detect_button)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.timeline_label = QLabel("疲劳/清醒时间轴（红=疲劳，绿=清醒，灰=未检测）：")
        layout.addWidget(self.timeline_label)
        self.timeline_widget = FatigueTimelineWidget(window_seconds=120.0)
        layout.addWidget(self.timeline_widget)

        tip = QLabel(
            "\n使用步骤：\n"
            "1. 等待模型预加载完成\n"
            "2. 选择检测功能\n"
            "3. 图片模式下选择图片并检测\n"
            "4. 摄像头模式下选择摄像头和帧率\n"
            "5. 在右侧查看画面、结果或对比表"
        )
        tip.setStyleSheet("color: #666;")
        layout.addStretch(1)
        layout.addWidget(tip)
        return group

    def build_result_panel(self) -> QGroupBox:
        group = QGroupBox("检测结果")
        layout = QVBoxLayout(group)

        self.image_label = QLabel("图片预览")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 360)
        self.image_label.setStyleSheet("border: 2px solid #ddd; background-color: #f5f5f5;")
        layout.addWidget(self.image_label, 2)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("检测结果将显示在这里。")
        layout.addWidget(self.result_text, 1)
        return group

    def start_preload(self) -> None:
        self.preload_label.setText("模型预加载：正在后台加载 5 种模型，首次启动可能需要较长时间...")
        self.detect_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.preload_worker = PreloadThread(self.server)
        self.preload_worker.finished.connect(self.on_preload_finished)
        self.preload_worker.start()

    def on_preload_finished(self, result: dict) -> None:
        self.progress.setVisible(False)
        self.preload_done = bool(result.get("success"))
        status = result.get("status") or []
        ok_count = sum(1 for item in status if item.get("success"))
        if self.preload_done:
            self.preload_label.setText(f"模型预加载：完成（{ok_count}/{len(status)}）")
            self.preload_label.setStyleSheet("color: #2e7d32;")
        else:
            self.preload_label.setText(f"模型预加载：失败 - {result.get('error', '未知错误')}")
            self.preload_label.setStyleSheet("color: #c62828;")
        self.update_action_state()

    def update_action_state(self) -> None:
        mode = self.mode_combo.currentText()
        if self.webcam_running:
            self.detect_button.setEnabled(True)
            return
        if mode == "实时摄像头检测":
            self.detect_button.setEnabled(self.preload_done and "YOLOv11 Classification" in self.available_methods)
        else:
            self.detect_button.setEnabled(self.preload_done and self.current_image_path is not None)

    def on_mode_changed(self, mode: str) -> None:
        if self.webcam_running:
            self.stop_webcam_detection()
        webcam = mode == "实时摄像头检测"
        comparison = mode == "五模型对比检测"
        self.select_button.setEnabled(not webcam)
        self.method_combo.setEnabled(not comparison and not webcam)
        self.camera_label.setEnabled(webcam)
        self.camera_combo.setEnabled(webcam and not self.webcam_running)
        self.fps_label.setEnabled(webcam)
        self.fps_combo.setEnabled(webcam and not self.webcam_running)
        self.timeline_label.setVisible(webcam)
        self.timeline_widget.setVisible(webcam)
        if webcam:
            self.detect_button.setText("启动摄像头检测")
            self.path_label.setText("摄像头实时模式：使用 YOLOv11 Classification")
            if "YOLOv11 Classification" not in self.available_methods:
                self.result_text.setHtml("<p style='color:red'>实时摄像头检测需要 YOLOv11 Classification 模型文件。</p>")
            else:
                self.result_text.setHtml("<p>实时摄像头检测会调用电脑摄像头，用人脸检测框出人脸，并用 YOLOv11 Classification 实时判断疲劳状态。</p>")
        elif comparison:
            self.detect_button.setText("开始五模型对比检测")
            self.result_text.setHtml("<p>五模型对比检测会依次调用所有可用方法，并在右侧汇总结果。</p>")
        else:
            self.detect_button.setText("开始单模型检测")
            self.result_text.clear()
        self.update_action_state()

    def select_image(self) -> None:
        dataset_dir = BASE_DIR / "DriverFatigueDetection" / "dataset_split" / "test"
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(dataset_dir if dataset_dir.exists() else BASE_DIR),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not image_path:
            return
        self.current_image_path = image_path
        self.path_label.setText(image_path)
        self.update_action_state()
        self.show_image(image_path)
        self.result_text.clear()

    def show_image(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("无法预览图片")
            return
        scaled = pixmap.scaled(
            self.image_label.width() - 20,
            self.image_label.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def show_frame(self, frame_bgr) -> None:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.image_label.width() - 20,
            self.image_label.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def start_detection(self) -> None:
        mode = self.mode_combo.currentText()
        if mode == "实时摄像头检测":
            if self.webcam_running:
                self.stop_webcam_detection()
            else:
                self.start_webcam_detection()
            return
        if not self.preload_done:
            QMessageBox.warning(self, "提示", "模型仍在预加载，请稍候。")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先选择图片。")
            return
        comparison = mode == "五模型对比检测"
        if comparison:
            methods = list(self.available_methods)
            if not methods:
                QMessageBox.warning(self, "提示", "没有可用的检测方法。")
                return
            waiting_text = "<p>正在使用已加载模型进行五模型对比检测，请稍候...</p>"
        else:
            method = self.method_combo.currentText()
            if not method:
                QMessageBox.warning(self, "提示", "请先选择检测方法。")
                return
            methods = [method]
            waiting_text = "<p>正在使用已加载模型检测，请稍候...</p>"

        self.select_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.result_text.setHtml(waiting_text)

        self.worker = PredictionThread(self.server, methods, self.current_image_path, comparison=comparison)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.start()

    def on_detection_finished(self, result: dict) -> None:
        self.progress.setVisible(False)
        self.select_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.method_combo.setEnabled(self.mode_combo.currentText() != "五模型对比检测")
        self.update_action_state()
        self.render_result(result)

    def start_webcam_detection(self) -> None:
        if not self.preload_done:
            QMessageBox.warning(self, "提示", "模型仍在预加载，请稍候。")
            return
        if "YOLOv11 Classification" not in self.available_methods:
            QMessageBox.warning(self, "提示", "未找到 YOLOv11 Classification 模型文件，无法启动实时摄像头检测。")
            return
        camera_index = int(self.camera_combo.currentText())
        target_fps = int(self.fps_combo.currentText())
        self.webcam_running = True
        self.last_webcam_text_update = 0.0
        self.timeline_widget.reset()
        self.detect_button.setText("停止摄像头检测")
        self.select_button.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.fps_combo.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.result_text.setHtml("<p>正在启动摄像头并加载 YOLOv11 模型，请稍候...</p>")
        self.webcam_worker = WebcamThread(self.server, camera_index, target_fps)
        self.webcam_worker.frame_ready.connect(self.on_webcam_frame)
        self.webcam_worker.result_ready.connect(self.on_webcam_result)
        self.webcam_worker.status_changed.connect(self.on_webcam_status)
        self.webcam_worker.stopped.connect(self.on_webcam_stopped)
        self.webcam_worker.start()

    def stop_webcam_detection(self) -> None:
        if self.webcam_worker is not None:
            self.webcam_worker.stop()
        self.detect_button.setEnabled(False)
        self.detect_button.setText("正在停止摄像头...")

    def on_webcam_frame(self, frame) -> None:
        self.show_frame(frame)

    def on_webcam_status(self, status: str) -> None:
        self.preload_label.setText(f"摄像头检测：{status}")
        self.preload_label.setStyleSheet("color: #0057b8;")

    def on_webcam_result(self, result: dict) -> None:
        if result.get("error") and not self.webcam_running:
            return
        prediction = result.get("prediction") or {}
        if result.get("success") and "is_drowsy" in prediction:
            self.timeline_widget.add_sample(bool(prediction.get("is_drowsy")))
        else:
            self.timeline_widget.add_sample(None)
        now = time.perf_counter()
        serious_error = bool(result.get("error")) and result.get("face_found") is not False
        if now - self.last_webcam_text_update >= 0.35 or serious_error:
            self.last_webcam_text_update = now
            self.render_webcam_result(result)

    def on_webcam_stopped(self) -> None:
        self.webcam_running = False
        self.webcam_worker = None
        self.progress.setVisible(False)
        self.mode_combo.setEnabled(True)
        webcam = self.mode_combo.currentText() == "实时摄像头检测"
        self.select_button.setEnabled(not webcam)
        self.method_combo.setEnabled(not webcam and self.mode_combo.currentText() != "五模型对比检测")
        self.camera_combo.setEnabled(webcam)
        self.fps_combo.setEnabled(webcam)
        self.detect_button.setText("启动摄像头检测" if webcam else "开始单模型检测")
        self.update_action_state()
        if self.preload_done:
            self.preload_label.setText("模型预加载：完成")
            self.preload_label.setStyleSheet("color: #2e7d32;")

    def render_webcam_result(self, result: dict) -> None:
        html = ["<h2>实时摄像头检测</h2>"]
        html.append(f"<p><b>目标帧率：</b>{result.get('target_fps', '-')} FPS</p>")
        actual_fps = result.get("actual_fps")
        if isinstance(actual_fps, (int, float)):
            html.append(f"<p><b>实际处理帧率：</b>{actual_fps:.2f} FPS</p>")
        html.append(f"<p><b>人脸检测后端：</b>{result.get('face_detector', '-')}</p>")
        if result.get("estimated_face_box"):
            html.append("<p style='color:#c47f00'><b>提示：</b>检测器未初始化成功，当前使用画面中心估计框进行 YOLO 分类。</p>")
        detector_errors = result.get("face_detector_errors") or []
        if detector_errors:
            html.append("<details><summary>人脸检测器初始化/运行错误</summary><ul>")
            for error in detector_errors:
                html.append(f"<li>{error}</li>")
            html.append("</ul></details>")
        device = result.get("device")
        if device:
            html.append(f"<p><b>推理设备：</b>{device}</p>")
        timing = result.get("timing") or {}
        inference_ms = timing.get("inference_ms")
        if isinstance(inference_ms, (int, float)):
            html.append(f"<p><b>YOLO 推理耗时：</b>{inference_ms:.1f} ms</p>")

        if not result.get("success"):
            html.append(f"<p style='color:#c47f00'><b>状态：</b>{result.get('error', '等待检测结果')}</p>")
            html.append(f"<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>")
            self.result_text.setHtml("\n".join(html))
            return

        prediction = result.get("prediction") or {}
        label = prediction.get("label", "unknown")
        is_drowsy = bool(prediction.get("is_drowsy"))
        zh_label = "疲劳/困倦" if is_drowsy else "清醒/非疲劳"
        color = "#c62828" if is_drowsy else "#2e7d32"
        html.append(f"<h1 style='color:{color}'>结果：{zh_label} ({label})</h1>")
        confidence = prediction.get("confidence")
        if isinstance(confidence, (int, float)):
            html.append(f"<p><b>置信度：</b>{confidence:.4f} ({confidence * 100:.2f}%)</p>")
        face_box = result.get("face_box") or {}
        if face_box:
            html.append(
                "<p><b>人脸框：</b>"
                f"x={face_box.get('x')}, y={face_box.get('y')}, w={face_box.get('w')}, h={face_box.get('h')}</p>"
            )
        probabilities = prediction.get("probabilities") or {}
        if probabilities:
            html.append("<h3>类别概率</h3><ul>")
            for key, value in probabilities.items():
                html.append(f"<li>{key}: {value:.4f} ({value * 100:.2f}%)</li>")
            html.append("</ul>")
        html.append("<h3>最近一次完整 JSON</h3>")
        html.append(f"<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>")
        self.result_text.setHtml("\n".join(html))

    def render_comparison_result(self, result: dict) -> None:
        image_path = result.get("image_path", "")
        results = result.get("results") or []
        html = ["<h2>五模型对比检测</h2>", f"<p><b>图片：</b>{image_path}</p>"]
        html.append(
            "<table border='1' cellspacing='0' cellpadding='6' style='border-collapse: collapse;'>"
            "<tr style='background-color:#f0f0f0;'>"
            "<th>检测方法</th><th>状态</th><th>检测结果</th><th>置信度/疲劳概率</th><th>类别概率</th><th>错误信息</th>"
            "</tr>"
        )
        success_count = 0
        drowsy_count = 0
        notdrowsy_count = 0
        for item in results:
            method = item.get("method", "未知方法")
            success = bool(item.get("success"))
            prediction = item.get("prediction") or {}
            label = prediction.get("label", "-")
            is_drowsy = bool(prediction.get("is_drowsy"))
            if success:
                success_count += 1
                if is_drowsy:
                    drowsy_count += 1
                else:
                    notdrowsy_count += 1
            zh_label = "疲劳/困倦" if is_drowsy else "清醒/非疲劳"
            if not success:
                zh_label = "-"
            confidence = prediction.get("confidence")
            prob_drowsy = prediction.get("prob_drowsy")
            score_parts = []
            if confidence is not None:
                score_parts.append(f"置信度 {confidence:.4f} ({confidence * 100:.2f}%)")
            if prob_drowsy is not None:
                score_parts.append(f"疲劳概率 {prob_drowsy:.4f} ({prob_drowsy * 100:.2f}%)")
            probabilities = prediction.get("probabilities") or {}
            prob_text = "<br>".join(
                f"{key}: {value:.4f} ({value * 100:.2f}%)" if isinstance(value, (int, float)) else f"{key}: {value}"
                for key, value in probabilities.items()
            )
            color = "#c62828" if is_drowsy else "#2e7d32"
            if not success:
                color = "#777"
            html.append(
                "<tr>"
                f"<td>{method}</td>"
                f"<td>{'成功' if success else '失败'}</td>"
                f"<td style='color:{color}; font-weight:bold'>{zh_label} ({label})</td>"
                f"<td>{'<br>'.join(score_parts) if score_parts else '-'}</td>"
                f"<td>{prob_text if prob_text else '-'}</td>"
                f"<td style='color:red'>{item.get('error', '') if not success else ''}</td>"
                "</tr>"
            )
        html.append("</table>")
        html.append(
            f"<h3>汇总</h3><p>成功方法数：{success_count}/{len(results)}；"
            f"判定疲劳：{drowsy_count}；判定清醒：{notdrowsy_count}</p>"
        )
        if success_count:
            final_text = "疲劳/困倦" if drowsy_count > notdrowsy_count else "清醒/非疲劳"
            final_color = "#c62828" if drowsy_count > notdrowsy_count else "#2e7d32"
            html.append(f"<h1 style='color:{final_color}'>多数投票结论：{final_text}</h1>")
        html.append("<h3>完整 JSON</h3>")
        html.append(f"<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>")
        self.result_text.setHtml("\n".join(html))

    def render_result(self, result: dict) -> None:
        if result.get("comparison"):
            self.render_comparison_result(result)
            return
        method = result.get("method", "未知方法")
        image_path = result.get("image_path", "")
        html = [f"<h2>{method}</h2>", f"<p><b>图片：</b>{image_path}</p>"]

        if not result.get("success"):
            html.append(f"<p style='color:red'><b>检测失败：</b>{result.get('error', '未知错误')}</p>")
            html.append(f"<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>")
            self.result_text.setHtml("\n".join(html))
            return

        prediction = result.get("prediction", {})
        label = prediction.get("label", "unknown")
        is_drowsy = bool(prediction.get("is_drowsy"))
        color = "#c62828" if is_drowsy else "#2e7d32"
        zh_label = "疲劳/困倦" if is_drowsy else "清醒/非疲劳"
        html.append(f"<h1 style='color:{color}'>结果：{zh_label} ({label})</h1>")

        confidence = prediction.get("confidence")
        if confidence is not None:
            html.append(f"<p><b>置信度：</b>{confidence:.4f} ({confidence * 100:.2f}%)</p>")
        prob_drowsy = prediction.get("prob_drowsy")
        if prob_drowsy is not None:
            html.append(f"<p><b>疲劳概率：</b>{prob_drowsy:.4f} ({prob_drowsy * 100:.2f}%)</p>")

        probabilities = prediction.get("probabilities") or {}
        if probabilities:
            html.append("<h3>类别概率</h3><ul>")
            for key, value in probabilities.items():
                if value is None:
                    html.append(f"<li>{key}: 无</li>")
                else:
                    html.append(f"<li>{key}: {value:.4f} ({value * 100:.2f}%)</li>")
            html.append("</ul>")

        features = result.get("features") or {}
        if features:
            html.append("<h3>关键特征</h3><ul>")
            count = 0
            for key, value in features.items():
                if count >= 12:
                    html.append("<li>更多特征见完整 JSON。</li>")
                    break
                if isinstance(value, (int, float)):
                    html.append(f"<li>{key}: {value:.4f}</li>")
                else:
                    html.append(f"<li>{key}: {value}</li>")
                count += 1
            html.append("</ul>")

        html.append("<h3>完整 JSON</h3>")
        html.append(f"<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>")
        self.result_text.setHtml("\n".join(html))

    def closeEvent(self, event) -> None:
        if self.webcam_worker is not None:
            self.webcam_worker.stop()
            self.webcam_worker.wait(3000)
        self.server.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
