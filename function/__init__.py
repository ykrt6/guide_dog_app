# 画像処理系
import cv2

# フィルター画像
img_danger = cv2.imread('static/ready/danger_filter.jpg', cv2.IMREAD_GRAYSCALE)
img_warn = cv2.imread('static/ready/warn_filter.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('static/ready/right_filter.jpg', cv2.IMREAD_GRAYSCALE)
img_left = cv2.imread('static/ready/left_filter.jpg', cv2.IMREAD_GRAYSCALE)

# ステータス状況の種類
place_lis = ['crosswalk', 'corner', 'stairs', 'other']

# ラベル
place_class_lis = ['crosswalk', 'rightcorner_u', 'leftcorner_u', 'rightcorner_l', 'leftcorner_l', 'stairs']	# 優先順 (これらのラベルは優先的に出力する)
border_lis = ['guardrail', 'shrubs', 'tree', 'fence']		# 車道と歩道の境界にあるもの
road_lis = ['car', 'motorbike', 'bus', 'truck']		# 車道にいるもの
obstacle_lis = ['person', 'bicycle', 'bicycler', 'traffic_light', 'handrail', 'bollard', 'pole', 'postbox', 'safety-cone']	# 歩道でぶつかる可能性があるもの (車道と歩道の境界にあるものは除く)
ground_lis = ['braille_block', 'white_line', 'steps']	# 歩道でぶつかる可能性がないもの

# 音声インターバル
sound_interval = 4

# 画像処理サイズ
process_img_size = 256
size = (process_img_size, process_img_size)