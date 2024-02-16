# 型変換系
import base64

# 画像処理系
import numpy as np
import cv2
from PIL import Image

# データ処理系？
from io import BytesIO


def base64_to_ndarray(img_base64:base64):
    """ 文字列 -> Numpy

    Args:
        img_base64 (base64): 画像データ
        save_size (tuple): 保存サイズ(幅、高さ)

    Returns:
        img (ndarry): 画像処理用
    
    """

    # binary <- string base64
    img_binary = base64.b64decode(img_base64.split(',')[1])
    # ipg <- binary
    img_jpg = np.frombuffer(img_binary, dtype=np.uint8)
    # raw image <- jpg
    img = cv2.imdecode(img_jpg, cv2.IMREAD_COLOR)

    return img


def pil_to_base64(img) -> base64:
    """ Pillow -> 文字列

    Args:
        img (pillow): 画像データ

    Returns:
        img_base64 (base64): webブラウザ表示用
    
    """

    img = Image.fromarray(img[:, :, ::-1])
    buffer = BytesIO()
    img.save(buffer, "jpeg")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return img_base64