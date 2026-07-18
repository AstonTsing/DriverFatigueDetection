"""训练 MobileNetV2 模型的独立脚本"""
import os
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, '..', 'DriverFatigueDetection', 'dataset_split')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')

# 超参数
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MOBILENET_EPOCHS = 25
L2_REG = 1e-4

print('Python:', sys.executable)
print('TensorFlow:', tf.__version__)
print('数据集路径:', DATA_ROOT)

# 检查数据集
for name, path in [('train', TRAIN_DIR), ('val', VAL_DIR)]:
    assert os.path.isdir(path), f'缺少目录: {path}'
    n_d = sum(len(files) for _, _, files in os.walk(os.path.join(path, 'drowsy')))
    n_n = sum(len(files) for _, _, files in os.walk(os.path.join(path, 'notdrowsy')))
    print(f'{name}: drowsy={n_d}, notdrowsy={n_n}, total={n_d + n_n}')

# 计算类别权重
_n_d = sum(len(files) for _, _, files in os.walk(os.path.join(TRAIN_DIR, 'drowsy')))
_n_n = sum(len(files) for _, _, files in os.walk(os.path.join(TRAIN_DIR, 'notdrowsy')))
_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array([0] * _n_d + [1] * _n_n))
CLASS_WEIGHT = {0: float(_weights[0]), 1: float(_weights[1])}
print('class_weight:', CLASS_WEIGHT)

# 数据生成器
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)
val_gen = eval_datagen.flow_from_directory(
  VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
  shuffle=False,
)

print('类别映射:', train_gen.class_indices)

# 构建 MobileNetV2 模型
print('\n开始构建 MobileNetV2 模型...')
mobilenet_base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
)
mobilenet_base.trainable = False

model_mobilenet = Sequential([
    mobilenet_base,
    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dense(512, activation='relu', kernel_regularizer=l2(L2_REG)),
    Dense(2, activation='softmax'),
])
model_mobilenet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
  metrics=['accuracy'],
)
model_mobilenet.summary()

# 训练模型
print(f'\n开始训练 (epochs={MOBILENET_EPOCHS})...')
history_mobilenet = model_mobilenet.fit(
    train_gen,
    epochs=MOBILENET_EPOCHS,
    validation_data=val_gen,
    class_weight=CLASS_WEIGHT,
    verbose=1,
)

# 保存模型
output_path = os.path.join(PROJECT_ROOT, 'model_mobilenetv2.h5')
model_mobilenet.save(output_path)
print(f'\n模型已保存: {output_path}')
