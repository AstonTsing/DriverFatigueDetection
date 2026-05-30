from __future__ import annotations

FEATURE_SCHEMA_VERSION = "v1_feature_schema_001"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 13, 291, 14]

FACE_LEFT = 234
FACE_RIGHT = 454
FOREHEAD = 10
CHIN = 152
NOSE_TIP = 1
NOSE_BRIDGE = 168
LEFT_BROW = [70, 63, 105]
RIGHT_BROW = [336, 296, 300]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
MOUTH_CORNERS = [61, 291]
MOUTH_VERTICAL = [13, 14]

FEATURE_NAMES = [
    "ear_left",
    "ear_right",
    "ear_mean",
    "mar",
    "ear_min",
    "ear_max",
    "ear_diff_abs",
    "ear_ratio_left_right",
    "left_eye_width_norm",
    "right_eye_width_norm",
    "left_eye_height_norm",
    "right_eye_height_norm",
    "mouth_width_norm",
    "mouth_height_norm",
    "mouth_open_area_proxy",
    "mouth_to_face_width",
    "mar_to_ear_mean",
    "mar_minus_ear_mean",
    "face_width",
    "face_height",
    "face_aspect_ratio",
    "nose_x_norm",
    "nose_y_norm",
    "nose_to_chin_norm",
    "forehead_to_chin_norm",
    "left_right_face_width_ratio",
    "eye_line_angle",
    "mouth_line_angle",
    "left_brow_eye_distance_norm",
    "right_brow_eye_distance_norm",
    "brow_eye_distance_mean",
    "brow_eye_distance_diff_abs",
    "brightness",
    "contrast",
    "blur_score",
]
