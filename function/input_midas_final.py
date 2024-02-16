import cv2
import numpy as np
from function.midas_load_2 import default_models
from PIL import Image

# img = "kyaribure/IMG_4970.jpg"
imgname =  "color_"
output_path = "./output_hikaku2/"


model_type = "dpt_hybrid_384"
model_weights = default_models[model_type]
optimize = False
height = None
square = False
grayscale = False
depth_min = 0
depth_max = 0


# 距離定義
fig_x = 4284
fig_y = 5712

# 入力画像の読み込み
# fig = Image.open(img)

# width,stature  = fig.size

# 390から360の中間
# x = 2075
# y = 3793

# 相対的な座標
y_coords = [4241 ,4083 ,3954 ,3840 ,3746 ,3671 ,3606 ,3548 ,3498 ,3456]
# y_normalized = [int(coord / fig_y * stature) for coord in y_coords]

# 絶対距離のリスト
absolute_distances = np.array([270,300,330,360,390,420,450,480,510,540])

# del fig
# del width
# del stature

"""
% Table generated by Excel2LaTeX from sheet 'Sheet1'
\begin{table}[htbp]
  \centering
  \caption{Add caption}
    \begin{tabular}{rrl}
    \multicolumn{1}{l}{IMG4970} & \multicolumn{1}{l}{y座標} & x座標 \\
    120   &       &  \\
    150   & 5666  & (2092.142857142857, \\
    180   & 5109  & (2092.142857142857, \\
    210   & 4721  & (2092.142857142857, \\
    240   & 4446  & (2092.142857142857, \\
    270   & 4241  & (2092.142857142857, \\
    300   & 4083  & (2075.7142857142853, \\
    330   & 3954  & (2075.7142857142853, \\
    360   & 3840  & (2075.7142857142853, \\
    390   & 3746  & (2075.7142857142853, \\
    420   & 3671  & (2075.7142857142853, \\
    450   & 3606  & (2075.7142857142853, \\
    480   & 3548  & (2075.7142857142853, \\
    510   & 3498  & (2075.7142857142853, \\
    540   & 3456  & (2070.0, \\
    570   & 3419  & (2068.5714285714284, \\
    600   & 3381  & (2068.5714285714284, \\
    630   & 3351  & (2068.5714285714284, \\
    660   & 3321  & (2064.7058823529405, \\
    690   & 3297  & (2067.0588235294113, \\
    720   & 3272  & (2118.8235294117644, \\
    750   & 3248  & (2074.117647058823, \\
    780   & 3228  & (2074.7058823529405, \\
    810   & 3209  & (2065.882352941176, \\
    840   & 3192  & (2065.882352941176, \\
    870   & 3177  & (2065.882352941176, \\
    900   & 3161  & (2066.9999999999995, \\
    930   & 3147  & (2066.9999999999995, \\
    \end{tabular}%
  \label{tab:addlabel}%
\end{table}%


"""
