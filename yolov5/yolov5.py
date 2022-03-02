import torch
import torch.backends.cudnn as cudnn
import os

from functools import lru_cache
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync

class YoloV5:
    def __init__(self):
        cwd = os.getcwd()
        weights_path = cwd + "/yolov5/weights/best.pt"
        yaml_path = cwd + "/yolov5/weights/data.yaml"
        self.weights = weights_path  # model.pt path(s)
        self.im = None
        self.im0s = None
        self.data=yaml_path  # dataset.yaml path
        self.imgsz=[640, 640]  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes=None # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.half=False  # use FP16 half-precision inference
        self.dnn=False
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        

    @torch.no_grad()
    def get_bounding_boxes(self, im, im0s):  # use OpenCV DNN for ONNX inference)


        stride, pt, jit, onnx, engine = self.model.stride, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.self.model.half() if self.half else self.model.self.model.float()

        bs = 1  # batch_size

        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=self.half)  # warmup
        dt = [0.0, 0.0, 0.0]
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = self.model(im, augment=self.augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        # Pred es una lista de predicciones por imagen
        pred_dict = {}
        det = pred[0]
        # Corre una sola vez, puesto que solo la corremos sobre una imagen

        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            #Ahora sí itera sobre las detecciones
            # Write results
            for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 
                x_start = int(xyxy[0])
                y_start = int(xyxy[1])
                x_end = int(xyxy[2])
                y_end = int(xyxy[3])
                pred_dict[i] = {
                    "class" : int(cls),
                    "bounding_box" : {
                        "x_start" : x_start,
                        "y_start" : y_start,
                        "x_end" : x_end,
                        "y_end" : y_end,
                        
                    },
                    "confidence" : float(conf),
                }
            return pred_dict

@lru_cache()
def get_yolov5():
    print("modelo yolo cargado")
    return YoloV5()


YoloV5 = get_yolov5()