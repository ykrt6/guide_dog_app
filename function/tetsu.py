from ultralytics import YOLO
import random
import numpy as np
import torch
import cv2

def init_obj_model():
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
    
    # Load a pretrained vidvipo_yolov8 model
    # model = YOLO('weights/vidvipo_yolov8.pt', task='detect')
    model = YOLO('weights/best_178.pt', task='detect')
    # model = YOLO('weights/yolov8n_best_16.pt', task='detect')

    predictor, is_cli = model.premodel(save=False, conf=0.3, save_txt=False, classes=[0,1,2,3,4,5,6,9,11,17,19,22,24,31,34,35,36,37,38,39,40,41,42])

    return model, predictor, is_cli

@torch.inference_mode()
def obj_detect(model, predictor, is_cli, img):
    # Run inference
    results = model.predict(predictor=predictor, is_cli=is_cli, source=img)

    return results[0].plot(), results[0].todict()


if __name__ == "__main__":
    model, predictor, is_cli = init_obj_model()

    img = cv2.imread('static/ready/sample_video_img_000.jpg')

    object, object_dict = obj_detect(model, predictor, is_cli, img)

    cv2.imwrite("test.jpg", object)