from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.regularizers import l2

base = Path(__file__).resolve().parent
model_path = base / 'Introduction_to_AI-Project_chenbh' / 'model_mobilenetv2.h5'
print('TF', tf.__version__)

mobilenet_base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
mobilenet_base.trainable = False
model = Sequential([
    mobilenet_base,
    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(2, activation='softmax'),
])
print('built')
try:
    model.load_weights(str(model_path))
    print('load_weights ok')
except Exception as e:
    print('load_weights failed', type(e).__name__, e)
try:
    model.load_weights(str(model_path), by_name=True, skip_mismatch=True)
    print('load_weights by_name ok')
except Exception as e:
    print('load_weights by_name failed', type(e).__name__, e)
