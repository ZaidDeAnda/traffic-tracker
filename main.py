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
from sort import *

import warnings

warnings.filterwarnings("ignore")

def main_func():
    output_path = "Output.mp4"

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture("/home/zaid/github/traffic-tracker/petal_20220217_160339.mp4")
    time.sleep(2.0)

    # initialize the video writer (we'll instantiate later if need be)
    writer = None
    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None
    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.5)
    trackableObjects = {}
    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    # start the frames per second throughput estimator
    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1]
        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, 30,
                (W, H), True)
        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = np.zeros([6])
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        status = "Detecting"
        im, im0s = ImageLoader.PreprocessImage(rgb)
        predictions = YoloV5.get_bounding_boxes(im, im0s)

        for prediction in predictions:
            if predictions[prediction]["confidence"] > 0.5:
                x_start, y_start, x_end, y_end = predictions[prediction]["bounding_box"].values()
                array = np.array([x_start, y_start, x_end, y_end, predictions[prediction]["confidence"], predictions[prediction]["class"]])
                rects = np.vstack((rects, array))
        rects = np.delete(rects, 0, 0)
        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = mot_tracker.update(rects)
        #loop over the tracked objects
        for (x_start, y_start, x_end, y_end, objectID, class_id) in objects:
            centroid_x = int((x_end + x_start)//2)
            centroid_y = int((y_end + y_start)//2)
            centroid = [centroid_x, centroid_y]
            print(centroid)
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)
            # if there is no existing trackable object, create one
            if to is None:
                # print(f"nuevo registro {objectID}")
                to = TrackableObject(objectID, centroid, class_id)
            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # print(f"Continuando registro de {objectID}")
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True
            # store the trackable object in our dictionary
            trackableObjects[objectID] = to
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {} + CLASS {}".format(objectID, class_id)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)
        # # show the output frame
        cv2.imshow("Frame", frame)
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
