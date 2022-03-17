from centroid_tracker import CentroidTracker
from trackable_object import TrackableObject
from yolov5.yolov5 import YoloV5
from yolov5.utils.image_loader import ImageLoader
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.general import xyxy2xywh
from yolov5.utils.plots import Annotator
from yolov5.utils.torch_utils import select_device

import warnings

warnings.filterwarnings("ignore")

def main_func(
        path_to_video : str, 
        deep_sort_model = "osnet_x0_25", 
        device = "cpu",
        output_path = "Output.mp4",
        config_deepsort = "deep_sort/configs/deep_sort.yaml"
    ):

    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    device = select_device(device)
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(path_to_video)
    time.sleep(2.0)
    writer = None
    W = None
    H = None
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )
    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    fps = FPS().start()
    names = {
        0 : "car",
        1 : "bus",
        2 : "truck",
        3 : "person",
        4 : "motorbike"
    }
    while True:
        frame = vs.read()
        frame = frame[1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, 30,
                (W, H), True)
        im, im0s = ImageLoader.PreprocessImage(rgb)
        predictions = YoloV5.get_bounding_boxes(im, im0s)
        annotator = Annotator(im0s, line_width=2, pil=not ascii)
        #agregar aquí que solo detecte como BB's aquellas dentro de cierta zona
        xywhs = xyxy2xywh(predictions[:, 0:4])
        confs = predictions[:, 4]
        clss = predictions[:, 5]
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0s)

        # draw boxes for visualization
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):

                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                if names[c] == "car":
                    color = (0,255,0)
                elif names[c] == "bus":
                    color = (255,0,0)
                elif names[c] == "truck":
                    color = (0,0,255)
                elif names[c] == "person":
                    color = (255,255,0)
                elif names[c] == "motorbike":
                    color = (0,255,255)
                annotator.box_label(bboxes, label, color=color)
        im0s = annotator.result()
        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)
        # # show the output frame
        #cv2.imshow("Frame", im0s)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        print(f"frame #{totalFrames}")
        fps.update()
        # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    # if we are not using a video file, stop the camera video stream
    vs.stop()
    # close any open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_func()
