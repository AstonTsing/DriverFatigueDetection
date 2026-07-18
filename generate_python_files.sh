#!/bin/bash
# 生成predictor.py和detection_gui.py的脚本

cd "$(dirname "$0")"

echo "生成 predictor.py..."
cat > predictor.py << 'PYEOF'
"""驾驶员疲劳检测 - 预测模块"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v0'))
sys.path.insert(0, str(BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v1'))


def predict_v0_svm(image_path):
    try:
        import joblib
        from src.landmark_features import FaceLandmarkFeatureExtractor, read_image
        from src.dataset import LABEL_TO_CLASS
        from src.rule_detector import RuleConfig

        model_path = BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v0' / 'models' / 'svm_ear_mar.joblib'
        config_path = BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v0' / 'config' / 'rule_config.json'

        model = joblib.load(model_path)
        config = RuleConfig.from_json(str(config_path))
        image = read_image(str(image_path))

        with FaceLandmarkFeatureExtractor(config.min_detection_confidence, config.min_tracking_confidence) as ext:
          feat = ext.extract(image)

        if not feat.face_found:
          return {'method': 'V0-SVM', 'success': False, 'error': '未检测到人脸'}

        X = [[feat.ear_left, feat.ear_right, feat.ear_mean, feat.mar]]
        pred = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0]

     return {
            'method': 'DriverFatigueDetection-v0 (SVM)',
            'success': True,
            'image_path': str(image_path),
     'prediction': {
             'label': LABEL_TO_CLASS[pred],
                'is_drowsy': bool(pred),
         'confidence': float(prob[pred]),
                'probabilities': {'notdrowsy': float(prob[0]), 'drowsy': float(prob[1])}
            },
        'features': {'ear_left': feat.ear_left, 'ear_right': feat.ear_right, 'ear_mean': feat.ear_mean, 'mar': feat.mar}
        }
    except Exception as e:
        return {'method': 'V0-SVM', 'success': False, 'error': str(e)}


def predict_v1_hgb(image_path):
    try:
        import joblib
        import json
        from src.feature_extractor import RichFeatureExtractor
        from src.image_io import read_image
        from src.dataset import LABEL_TO_CLASS
        from src.metrics import predict_prob_drowsy

    model_path = BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v1' / 'models' / 'hgb_fatigue.joblib'
        config_path = BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v1' / 'config' / 'model_config.json'

        bundle = joblib.load(model_path)
        model = bundle['pipeline']
        with open(config_path) as f:
            cfg = json.load(f)

        image = read_image(str(image_path))
        with RichFeatureExtractor(float(cfg.get('min_detection_confidence', 0.5)), float(cfg.get('min_tracking_confidence', 0.5))) as ext:
            res = ext.extract(image)

     if not res.face_found:
            return {'method': 'V1-HGB', 'success': False, 'error': '未检测到人脸'}

        X = [res.feature_vector()]
        pred = int(model.predict(X)[0])
        prob_d = predict_prob_drowsy(model, X)
        prob = model.predict_proba(X)[0]

        return {
            'method': 'DriverFatigueDetection-v1 (HistGradientBoosting)',
            'success': True,
            'image_path': str(image_path),
            'prediction': {
                'label': LABEL_TO_CLASS[pred],
          'is_drowsy': bool(pred),
                'prob_drowsy': float(prob_d[0]),
              'probabilities': {'notdrowsy': float(prob[0]), 'drowsy': float(prob[1])}
          },
          'features': res.features
        }
    except Exception as e:
        return {'method': 'V1-HGB', 'success': False, 'error': str(e)}


def predict_yolo(image_path):
    try:
        from ultralytics import YOLO
     model_path = BASE_DIR / 'drowsiness_detection' / 'outputs' / 'test1' / 'best.pt'
        model = YOLO(str(model_path))
        results = model.predict(str(image_path), imgsz=224, verbose=False)
        res = results[0]

        pred_idx = int(res.probs.top1)
        pred_name = res.names[pred_idx]
        conf = float(res.probs.top1conf)
        probs_data = res.probs.data.cpu().numpy()

        return {
          'method': 'YOLOv11 Classification',
          'success': True,
          'image_path': str(image_path),
       'prediction': {
            'label': pred_name,
                'is_drowsy': pred_name == 'drowsy',
              'confidence': conf,
           'probabilities': {res.names[i]: float(probs_data[i]) for i in range(len(res.names))}
            }
      }
    except Exception as e:
        return {'method': 'YOLO', 'success': False, 'error': str(e)}


def predict_tf_baseline(image_path):
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        model_path = BASE_DIR / 'Introduction_to_AI-Project_chenbh' / 'model_baseline.h5'
        model = tf.keras.models.load_model(str(model_path))

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        pred_prob = model.predict(img, verbose=0)[0]
        pred_idx = int(np.argmax(pred_prob))
        labels = ['drowsy', 'notdrowsy']
        pred_label = labels[pred_idx]

        return {
            'method': 'TensorFlow Baseline CNN',
            'success': True,
            'image_path': str(image_path),
            'prediction': {
                'label': pred_label,
              'is_drowsy': pred_label == 'drowsy',
                'confidence': float(pred_prob[pred_idx]),
              'probabilities': {labels[i]: float(pred_prob[i]) for i in range(len(labels))}
            }
        }
    except Exception as e:
        return {'method': 'TF-Baseline', 'success': False, 'error': str(e)}


def predict_tf_mobilenet(image_path):
    try:
    import tensorflow as tf
        import cv2
        import numpy as np
        model_path = BASE_DIR / 'Introduction_to_AI-Project_chenbh' / 'model_mobilenetv2.h5'
     model = tf.keras.models.load_model(str(model_path))

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
      img = img.astype(np.float32) / 255.0
     img = np.expand_dims(img, 0)

        pred_prob = model.predict(img, verbose=0)[0]
        pred_idx = int(np.argmax(pred_prob))
        labels = ['drowsy', 'notdrowsy']
        pred_label = labels[pred_idx]

        return {
     'method': 'TensorFlow MobileNetV2',
            'success': True,
            'image_path': str(image_path),
            'prediction': {
             'label': pred_label,
          'is_drowsy': pred_label == 'drowsy',
                'confidence': float(pred_prob[pred_idx]),
                'probabilities': {labels[i]: float(pred_prob[i]) for i in range(len(labels))}
            }
        }
    except Exception as e:
        return {'method': 'TF-MobileNet', 'success': False, 'error': str(e)}


PREDICTORS = {
    'DriverFatigueDetection-v0 (SVM)': (predict_v0_svm, BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v0' / 'models' / 'svm_ear_mar.joblib'),
    'DriverFatigueDetection-v1 (HistGradientBoosting)': (predict_v1_hgb, BASE_DIR / 'DriverFatigueDetection' / 'DriverFatigueDetection-v1' / 'models' / 'hgb_fatigue.joblib'),
    'YOLOv11 Classification': (predict_yolo, BASE_DIR / 'drowsiness_detection' / 'outputs' / 'test1' / 'best.pt'),
    'TensorFlow Baseline CNN': (predict_tf_baseline, BASE_DIR / 'Introduction_to_AI-Project_chenbh' / 'model_baseline.h5'),
    'TensorFlow MobileNetV2': (predict_tf_mobilenet, BASE_DIR / 'Introduction_to_AI-Project_chenbh' / 'model_mobilenetv2.h5'),
}


def get_available_methods():
    available = []
    for name, (func, model_path) in PREDICTORS.items():
        if model_path.exists():
        available.append(name)
    return available


def predict_with_method(method_name, image_path):
    if method_name in PREDICTORS:
        func, _ = PREDICTORS[method_name]
     return func(image_path)
    return {'method': method_name, 'success': False, 'error': '未知方法'}


if __name__ == '__main__':
    print('可用方法:', get_available_methods())
PYEOF

echo "✓ predictor.py 已生成"

# 此脚本只生成predictor.py
# detection_gui.py由于较长将单独处理
