import torch
import cv2
import random

import numpy as np
from function.midas_load_2 import default_models, load_model

from logging import getLogger, StreamHandler, Formatter, DEBUG, ERROR
# loggin 設定
formatter = Formatter('[%(levelname)s %(asctime)s - %(message)s]')
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(formatter)
logger.setLevel(DEBUG)
logger.addHandler(handler)

def init_dep_model():
    model_weights = None
    model_type = "dpt_hybrid_384"
    optimize = False
    height = None
    square = False

    # args = parser.parse_args()
    if model_weights is None:
        model_weights = default_models[model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    # parserなどで指定
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed)

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s" % device)

    model, transform, net_w, net_h = load_model(
        device, model_weights, model_type, optimize, height, square
    )
    loaded_model = (model, transform, net_w, net_h)

    return device, loaded_model

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


def read_image(img):
    # img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def write_depth(depth, grayscale, bits=1):

    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    # depth_maxに対応するRGB値を取得します
    # color_mapped_pixel = cv2.applyColorMap(np.array([[depth_max]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]


    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_TURBO)
    if bits == 1:
        return out.astype("uint8")
    elif bits == 2:
        return out.astype("uint16")

@torch.inference_mode()
def depth_predict(img, device, loaded_model):
    """深度推定

    Args:
        img (pillow) : 画像データ
        device (?): cuda or cpu
        loaded_model (?): モデル

    Returns:
        out : 深度マップ
        prediction : 深度値 (全ピクセル)
        dep_info : 最大深度値、最小深度値 (削除済み)
    """
    model_type = "dpt_hybrid_384"
    optimize = False
    grayscale = False

    # 画像処理
    # start_time = time.time()

    # input
    original_image_rgb = read_image(img)  # in [0, 1]
    image = loaded_model[1]({"image": original_image_rgb})["image"]

    # compute
    with torch.no_grad():
        prediction = process(
            device, loaded_model[0], model_type, image, (loaded_model[2], loaded_model[3]), original_image_rgb.shape[1::-1], optimize
        )

    out = write_depth(prediction, grayscale, bits=2)
    # dep_info = (prediction.max(), prediction.min())
    
    # torch.cuda.empty_cache()

    return out, prediction

if __name__ == "__main__":
    device, loaded_model = init_dep_model()

    img = cv2.imread('static/ready/sample_video_img_000.jpg')

    depth, prediction, depth_max = depth_predict(img, device, loaded_model)