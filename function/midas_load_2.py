import cv2
import torch

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

from logging import getLogger, StreamHandler, Formatter, DEBUG, ERROR
# loggin 設定
formatter = Formatter('[%(levelname)s %(asctime)s - %(message)s]')
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(ERROR)
handler.setFormatter(formatter)
logger.setLevel(ERROR)
logger.addHandler(handler)

default_models = {
    "dpt_hybrid_384": "weights/dpt_hybrid_384.pt",
}


def load_model(device, model_path, model_type="dpt_hybrid_384", optimize=True, height=None, square=False):
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?

    Returns:
        The loaded network, the transform which prepares images as input to the network and the dimensions of the
        network input
    """
    

    keep_aspect_ratio = not square

    #model_type == "dpt_hybrid_384"
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        )
    net_w, net_h = 384, 384
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    
    logger.info("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))


    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    
    model.eval()

    model.to(device)

    return model, transform, net_w, net_h
