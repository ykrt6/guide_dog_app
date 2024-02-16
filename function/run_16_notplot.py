"""Compute depth maps for images in the input folder.
一枚にかかる時間が測れるバージョン
画像をストリーム処理(メモリに一度に読み込むのではなく必要なときに読み込む方法)
無駄なif文を削除
utilsの読み込みを廃止
midasフォルダの読み込み廃止(速度向上との費用対効果によりやらない)
写真ファイルの名前順による読み込み処理実装(写真を撮るプログラムから変数として与えられるため必要なしだが、これには実装済み)
写真を一枚だけ読み込むように変更
AWS用プログラムに対応
座標の入力に対応
絶対距離のフィッティング関数対応
単眼深度推定最終版
"""
import os
import torch
import cv2
import time
import datetime
import tkinter.simpledialog as simpledialog
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from function.midas_load_2 import load_model
from function.distance_evaluation_final import inverse_model, inverse
from function.midas_loss import ScaleAndShiftInvariantLoss

from function.input_midas_final import absolute_distances, output_path, model_type, model_weights,optimize,height,square,grayscale, imgname

#KMP_DUPLICATE_LIB_OK = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



#画像処理用の PyTorch モデルに対して画像を入力し、モデルの予測結果を取得し、指定されたサイズに拡大縮小した後に返す
def process(device, model, model_type, image, input_size, target_size, optimize):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction


#画像ファイルを読み込み、色空間の変換と正規化を行ってニューラルネットワークへの入力として適した形式に変換する
def read_image(img):
    img = cv2.imread(img)
    

    if img.ndim == 2 :
        img = cv2.cvtColor(img, cv2.COLOR_GRY2BGR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


#与えられた深度マップデータを画像として保存する。保存形式やカラーマップの適用に関する設定は、引数によって制御される
def write_depth(path, depth, grayscale, bits=1):

    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    print(f"最小深度{depth_min}")
    print(f"最大深度{depth_max}")

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_TURBO)
    if bits == 1:
        cv2.imwrite(path + ".jpg", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".jpg", out.astype("uint16"))


#与えられた深度マップ（depth）から特定の位置（x座標とy座標）の深度値を取得するための関数
def get_depth_at_position(depth, x, y):
    
    return depth[y, x]


#与えられた画像ファイルのサイズ（幅と高さ）を取得するための関数
def get_image_size(img):
    try:
        with Image.open(img) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None, None



def depth10(depth_value,depth_frame,*y_normalized):

    #深度値取得(for文回避)
    depth0 = depth_frame[y_normalized[0],:]
    d0 = np.min(depth0)

    depth1 = depth_frame[y_normalized[1],:]
    d1 = np.min(depth1)

    depth2 = depth_frame[y_normalized[2],:]
    d2 = np.min(depth2)

    depth3 = depth_frame[y_normalized[3],:]
    d3 = np.min(depth3)

    depth4 = depth_frame[y_normalized[4],:]
    d4 = np.min(depth4)

    depth5 = depth_frame[y_normalized[5],:]
    d5 = np.min(depth5)

    depth6 = depth_frame[y_normalized[6],:]
    d6 = np.min(depth6)

    depth7 = depth_frame[y_normalized[7],:]
    d7 = np.min(depth7)

    depth8 = depth_frame[y_normalized[8],:]
    d8 = np.min(depth8)

    depth9 = depth_frame[y_normalized[9],:]
    d9 = np.min(depth9)

    relative_values = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])  # 深度値のリスト
    # 結果を出力
    # print(relative_values)
    

    #逆関数
    inverseans_ans = inverse(d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,depth_value)
    


    return  inverseans_ans, relative_values



def depth_predict(img):
    
    #処理部#
    #CNNの設定
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 使用するデバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    #各種パラメータの設定
    model, transform, net_w, net_h = load_model(device, model_weights, model_type, optimize, height, square)

    #出力フォルダの作成
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)


    print("Start processing")
    # 画像を一枚だけ処理するために、以下の行をコメントアウトします
    # image_names = sorted(glob.glob(os.path.join(input_path, "*")))
    num_images = 1

    start_time = time.time()
    
    original_image_rgb = read_image(img)  # in [0, 1]
    image = transform({"image": original_image_rgb})["image"]

    # 推論
    with torch.no_grad():
        prediction = process(
            device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1], optimize
        )

    print(f"Image size: {prediction.shape[1]} x {prediction.shape[0]}")

    # X,Y座標の入力
    #x = yoloから入力 現在はinput_midasから入力中
    #y = yoloから入力 現在はinput_midasから入力中



    # X,Y座標の入力が画像の範囲内ならX,Y座標の深度値を出力する
    if 0 <= x < prediction.shape[1] and 0 <= y < prediction.shape[0]:
        depth_at_position = get_depth_at_position(prediction, x, y)
        print(f"深度値 ({x}, {y}): {depth_at_position:.4f}")

        #絶対距離計算
        distance= depth10(depth_at_position,prediction,y_normalized[0],y_normalized[1],y_normalized[2],y_normalized[3],y_normalized[4],y_normalized[5],y_normalized[6],y_normalized[7],y_normalized[8],y_normalized[9])

        print("inverse",distance[0][2],"cm")
        
        # 画像保存
        now = datetime.datetime.now()
        filename = output_path + imgname + now.strftime('_%Y%m%d_%H%M%S') + model_type 
        write_depth(filename, prediction, grayscale, bits=2)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.4f} seconds")

    else:
        print("範囲外")

    print("finish")



if __name__ == "__main__" :
    #while True:
        depth_predict(img)