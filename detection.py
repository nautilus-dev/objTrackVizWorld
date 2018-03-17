# USAGE
# python3 real_time_object_detection2.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import copy
from math import sqrt
import os
import iA

# Global variables (later with Gui)
def measure_dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)



def detect(input_path, output_path, conf = 0.5, distance=4, A_upper=3.5, A_lower=0.1, out=0.2, killtime=15):
    """conf für Wahrscheinichkeit, dist ist der Nenner für den Abstand der Schwerpunkte (1/dist mal die maximale
    Kantenlänge dard der Punkt springen), A_upper ist die obere Grenze für den Flächeninhalt, A_lower die untere,
    out für die äußeren 25prozent des Bildes, killtime ist die Anzahl der Frames bis ein objekt entfernt wird am Rand)
    """
    d = time.strftime("%d_%m_%Y")
    t = time.strftime("%H_%M_%S")
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(input_path)
    kmax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    vs = FileVideoStream(input_path).start()
    time.sleep(2.0)
    fps = FPS().start()
    k = 0
    track_obj = {}
    delater = []
    # Get the number of instances (not needed yet)
    for e in iA.instances:
        for i in iA.instances[e]:
            if e not in track_obj:
                track_obj[e] = 1
            else:
                track_obj[e] += 1
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        k += 1
        frame = imutils.resize(frame, width=xmax)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > conf:
                idx = int(detections[0, 0, i, 1])
                # look wether the obj should be tracked
                if CLASSES[idx] in track_obj:
                    helper = iA.iA(k)
                    iA.number_of_iA -= 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    helper.rect = [[startX, startY], [endX, endY]]
                    helper.calibrate()
                    # draw the prediction on the frame if in right dist
                    for e in iA.instances[CLASSES[idx]]:
                        # k1 = min(helper.A, e.A) / max(helper.A, e.A)
                        # get value per GUI?
                        k2 = max(abs(e.rect[0][0]-e.rect[1][0]), abs(e.rect[0][1]-e.rect[1][1])) / distance
                        # also maybe a predicted position can help (euler)
                        if measure_dist(helper.p, e.p) < k2 and e.detected == 0 and helper.A <= e.A*A_upper and helper.A >= e.A*A_lower and k >= e.tf:
                            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                            cv2.circle(frame, (int(e.p[0]),int(e.p[1])), 5, COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            if startX < 0:
                                startX = 0
                            if startY < 0:
                                startY = 0
                            if endX > xmax:
                                endX = xmax
                            if endY > ymax:
                                endY = ymax
                            cv2.putText(frame, label + " " + "I: " + str(e.number), (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            e.rect = [[startX, startY], [endX, endY]]
                            e.calibrate()
                            e.detected = 1
                            e.counter = 0
                            string = str(k) + "\t"+ "interestarea" + str(e.number) + "\t" +str(startX) + "\t" + str(startY) + "\t" + str(endX) + "\t" +  str(endY)
                            e.writer(string, output_path, d, t)
        for e in iA.instances:
            for i in iA.instances[e]:
                if i.detected == 0  and k >= i.tf:
                    startX = i.rect[0][0]
                    startY = i.rect[0][1]
                    endX = i.rect[1][0]
                    endY = i.rect[1][1]
                    # 100 to have an other color if detection is lost
                    cv2.rectangle(frame, (startX, startY), (endX, endY), 100, 2)
                    cv2.circle(frame, (int(i.p[0]),int(i.p[1])), 5, 100, 2)
                    i.rect = [[startX, startY], [endX, endY]]
                    i.calibrate()
                    string = str(k) +"\t"+"interestarea" + str(i.number) + "\t" +str(startX) + "\t" + str(startY) + "\t" + str(endX) + "\t" +  str(endY)
                    if (startY <= 3 or startX <= 3 or endX >= xmax - 3 or endY >= ymax - 3) and (i.p[0] < xmax * out or i.p[0] > xmax * (1-out)
                                                                                                 or i.p[1] < ymax * out or i.p[1] > ymax * (1-out)):
                        i.counter += 1
                        if i.counter >= killtime:
                            print("Instance " + str(i.number) +  " leaves the view frustum")
                            track_obj[e] -= 1
                            if track_obj[e] == 0:
                                del track_obj[e]
                            delater.append([e,i])
                    i.writer(string, output_path, d, t)
                i.detected = 0

        # show the output frame
        key = cv2.waitKey(1) & 0xFF
        h = 0
        l = len(delater)
        while h < l:
            # print(instances[delater[h][0]])
            # print(delater[h][1])
            iA.instances[delater[h][0]].remove(delater[h][1])
            if not iA.instances[delater[h][0]]:
                del iA.instances[delater[h][0]]
            if not iA.instances:
                print("No more Instances anymore")
                key = ord("q")
            h += 1
        # if the `q` key was pressed, break from the loop and clean the delater list for next frame
        delater = []
        procent = round((k / kmax) * 100,1)
        cv2.putText(frame, str(procent)+"%", (xmax-48, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 100, 2)
        cv2.imshow("Detection", frame)
        if key == ord("q") or k == kmax:
            break
        # update the FPS counter
        fps.update()
    # stop the timer and display FPS information
    fps.stop()
    if k == kmax:
        iA.compl_writer(output_path, d, t)
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    plant = iA.iA(0, [[91,5],[149,136]],"pottedplant" )
    stuhl = iA.iA(0, [[188, 101],[245,211]], "chair")
    p1 = iA.iA(180, [[295,3],[384,225]], "person")
    p2 = iA.iA(690, [[28,0],[125,226]], "person")
    xmax = 400
    ymax = round((xmax/16)*9)
    input_path = "hause.mp4"
    output_path = os.getcwd()
    detect(input_path, output_path)









    
