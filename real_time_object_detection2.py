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
from math import sqrt

# Global variables (later with Gui)
number_of_iA = 0
track_obj = {}
instances = {}
verbose = 0

class iA:
    def __init__(self):
        """
        A - ist the areas-value of the bb
        p - is the centerpoint
        rect - are the cooordinates of the actual bb
        detected - gives informations wether the instances was
        already detected in this frame
        number - is just for the writer (each instance has a number)
        """
        self.A = 0
        self.p = [0, 0]
        self.rect = [[0, 0], [0, 0]]
        self.c = ""
        self.calA()
        self.calp()
        self.detected = 0
        global number_of_iA
        number_of_iA += 1
        self.number = number_of_iA

    def calA(self):
        s1 =  max(self.rect[0][0], self.rect[1][0]) - min(self.rect[0][0], self.rect[1][0])
        s2 =  max(self.rect[0][1], self.rect[1][1]) - min(self.rect[0][1], self.rect[1][1])
        self.A = s1 * s2

    def calp(self):
        v1 = min(self.rect[0][0], self.rect[1][0])
        v2 = min(self.rect[0][1], self.rect[1][1])
        s1 =  max(self.rect[0][0], self.rect[1][0]) - v1
        s2 =  max(self.rect[0][1], self.rect[1][1]) - v2
        self.p = [v1 + 0.5 * s1, v2 + 0.5 * s2]

    def calibrate(self):
        self.calA()
        self.calp()

    def writer(self, string):
        writedata("interestarea_" + str(self.number), string) 


def writedata(name, string):
        obj = open(name + ".txt", "a")
        obj.write(string + "\n")
        obj.close()

def measure_dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = FileVideoStream("reiten.mkv").start()
#vs = FileVideoStream("netwon_2.mp4").start()
vs = FileVideoStream("shortcut_gorilla.avi").start()
time.sleep(2.0)
fps = FPS().start()
k = 0

# Track the horse

horse = iA()
horse.c = "horse"
instances[horse.c] = [horse]

# Track the person
person = iA()
person.c = "person"
instances[person.c] = [person]

# Get the number of instances (not needed yet)
for e in instances:
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
	frame = imutils.resize(frame, width=800)
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
	    if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                # verbose for first init (later we have to change this)
                if verbose < number_of_iA and CLASSES[idx] in track_obj:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    # 0 because there is just one obj. - change this later!
                    instances[CLASSES[idx]][0].rect = [[startX, endX], [startY, endY]]
                    instances[CLASSES[idx]][0].calibrate()
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    verbose += 1
                # look wether the obj should be tracked
                if CLASSES[idx] in track_obj:
                    helper = iA()
                    number_of_iA -= 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    helper.rect = [[startX, endX], [startY, endY]]
                    helper.calibrate()
                    # draw the prediction on the frame if in right dist
                    for e in instances[CLASSES[idx]]:
                        # k1 = min(helper.A, e.A) / max(helper.A, e.A)
                        # get value per GUI?
                        k2 = max(abs(e.rect[0][0]-e.rect[1][0]), abs(e.rect[0][1]-e.rect[1][1])) / 4
                        print(helper.p)
                        print(e.p)
                        print("----------------------")
                        # k1 < 2 and k1 > 0.5 misses
                        # also maybe a predicted position can help (euler)
                        if measure_dist(helper.p, e.p) < k2 and e.detected == 0:
                            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            e.rect = [[startX, endX], [startY, endY]]
                            e.calibrate()
                            e.detected = 1
                            string = str(k) + " "+CLASSES[idx] + " " +str(startX) + " " + str(startY) + " " + str(endX) + " " +  str(endY)
                            # writedata("test",string)
                            e.writer(string)
	for e in instances:
            for i in instances[e]:
                if i.detected == 0:
                    startX = i.rect[0][0]
                    startY = i.rect[1][0]
                    endX = i.rect[0][1]
                    endY = i.rect[1][1]
                    # 100 to have an other color if detection is lost
                    cv2.rectangle(frame, (startX, startY), (endX, endY), 100, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    i.rect = [[startX, endX], [startY, endY]]
                    i.calibrate()
                    string = str(k) + " "+i.c + " " +str(startX) + " " + str(startY) + " " + str(endX) + " " +  str(endY)
                    # writedata("test",string)
                    i.writer(string)
                i.detected = 0
                        

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
