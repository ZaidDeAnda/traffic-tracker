import numpy as np
from yolov5.utils.augmentations import letterbox

class ImageLoader():
    @classmethod
    def PreprocessImage(kls, img0):
        img_size=640
        stride=32
        auto=True
        # Padded resize
        img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img, img0