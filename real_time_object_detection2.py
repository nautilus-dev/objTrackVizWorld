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

# Global variables (later with Gui)
number_of_iA = 0
track_obj = {}
instances = {}
verbose = 0
delater = []

class iA:
    def __init__(self, trackframe=0, box=[[0,0],[0,0]], label= ""):
        """
        A - ist the areas-value of the bb
        p - is the centerpoint
        rect - are the cooordinates of the actual bb
        detected - gives informations wether the instances was
        already detected in this frame
        number - is just for the writer (each instance has a number)
        """
        self.A = 0
        self.Amax = 0
        self.p = [0, 0]
        self.rect = box
        self.c = label
        self.calibrate()
        self.detected = 0
        self.counter = 0
        self.tf = trackframe
        global number_of_iA
        number_of_iA += 1
        self.number = number_of_iA
        if self.c != "":
                 insert_inst(self.c, self)

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
        if self.A > self.Amax:
            self.Amax = self.A
        self.calp()

    def writer(self, string):
        writedata("interestarea_" + str(self.number), string) 


def writedata(name, string):
    """ For windwos
    if not os.path.exists(output_path + "\data"):
        os.makedirs(output_path + "\data")
    if not os.path.exists(output_path + "\data" + "\data_" + d + "_" +t):
        os.makedirs(output_path + "\data" + "\data_" + d + "_" +t)
    obj = open(output_path + "\data" + "\data_" + d + "_" +t + "\" + name + ".txt", "a")
    obj.write(string + "\n")
    obj.close()
    """
    if not os.path.exists(output_path + "/data"):
        os.makedirs(output_path + "/data")
    if not os.path.exists(output_path + "/data" + "/data_" + d + "_" +t):
        os.makedirs(output_path + "/data" + "/data_" + d + "_" +t)
    obj = open(output_path + "/data" + "/data_" + d + "_" +t + "/" + name + ".txt", "a")
    obj.write(string + "\n")
    obj.close()

def measure_dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def insert_inst(klasse, objekt):
    if klasse in instances:
        instances[klasse].append(objekt)
    else:
        instances[klasse] = [objekt]


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


d = time.strftime("%d_%m_%Y")
t = time.strftime("%H_%M_%S")

input_path = "helper.mp4"
input_path = "hause.mp4"
output_path = os.getcwd()
#vs = FileVideoStream("reiten.mkv").start()
#vs = FileVideoStream("netwon_2.mp4").start()
#vs = FileVideoStream("shortcut_gorilla.avi").start()
#vs = FileVideoStream("bb.mkv").start()
cap = cv2.VideoCapture(input_path)
kmax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

vs = FileVideoStream(input_path).start()
time.sleep(2.0)
fps = FPS().start()
k = 0
xmax = 400
ymax = round((xmax/16)*9)



# Track the cars from helper.mp4
"""
car1 = iA(0, [[170,44],[362,142]], "car")

car2 = iA(420, [[261,64],[388,132]], "car")

car3 = iA(900, [[38,76],[161,128]], "car")

"""
# Track zu Hause

plant = iA(0, [[91,5],[149,136]],"pottedplant" )
stuhl = iA(0, [[188, 101],[245,211]], "chair")

p1 = iA(180, [[295,3],[384,225]], "person")

p2 = iA(690, [[28,0],[125,226]], "person")


# Get the number of instances (not needed yet)
for e in instances:
    for i in instances[e]:
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
	#cv2.rectangle(frame, (188,101), (245, 211), 100, 2)

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
                """
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
                """
                # look wether the obj should be tracked
                if CLASSES[idx] in track_obj:
                    helper = iA(k)
                    number_of_iA -= 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    helper.rect = [[startX, startY], [endX, endY]]
                    helper.calibrate()
                    # draw the prediction on the frame if in right dist
                    for e in instances[CLASSES[idx]]:
                        # k1 = min(helper.A, e.A) / max(helper.A, e.A)
                        # get value per GUI?
                        k2 = max(abs(e.rect[0][0]-e.rect[1][0]), abs(e.rect[0][1]-e.rect[1][1])) / 4
                        # also maybe a predicted position can help (euler)
                        if measure_dist(helper.p, e.p) < k2 and e.detected == 0 and helper.A <= e.A*3.5 and helper.A >= e.A*0.1 and k >= e.tf:
                            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                            cv2.circle(frame, (int(e.p[0]),int(e.p[1])), 5, COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label + " " + "I: " + str(e.number), (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            e.rect = [[startX, startY], [endX, endY]]
                            e.calibrate()
                            e.detected = 1
                            e.counter = 0
                            string = str(k) + " "+CLASSES[idx] + " " +str(startX) + " " + str(startY) + " " + str(endX) + " " +  str(endY)
                            # writedata("test",string)
                            e.writer(string)
	for e in instances:
            for i in instances[e]:
                if i.detected == 0  and k >= i.tf:
                    startX = i.rect[0][0]
                    startY = i.rect[0][1]
                    endX = i.rect[1][0]
                    endY = i.rect[1][1]
                    #if max(startX, endX, startY, endY) == 400 or min(startX, endX, startY, endY) == 0:
                        #print(startX)
                        #print(endX)
                        #print(startY)
                        #print(endY)
                        #print("---------")
                    # 100 to have an other color if detection is lost
                    cv2.rectangle(frame, (startX, startY), (endX, endY), 100, 2)
                    cv2.circle(frame, (int(i.p[0]),int(i.p[1])), 5, 100, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    i.rect = [[startX, startY], [endX, endY]]
                    i.calibrate()
                    string = str(k) + " "+i.c + " " +str(startX) + " " + str(startY) + " " + str(endX) + " " +  str(endY)
                    a = startY + (endY - startY) / 2
                    b = startX + (endX - startX) / 2
                    if (startY <= 3 or startX <= 3 or endX >= xmax - 3 or endY >= ymax - 3) and (i.p[0] < xmax * 0.25 or i.p[0] > xmax * 0.75
                                                                                                 or i.p[1] < ymax * 0.25 or i.p[1] > ymax * 0.75):
                        i.counter += 1
                        if i.counter >= 10:
                            print("Instance " + str(i.number) +  " leaves the view frustum")
                            track_obj[e] -= 1
                            if track_obj[e] == 0:
                                del track_obj[e]
                            delater.append([e,i])
                    # writedata("test",string)
                    i.writer(string)
                    
                i.detected = 0

	# show the output frame
	#if delater:
            #print(k)
            #print(delater)
            #print("-------------")
	key = cv2.waitKey(1) & 0xFF
	h = 0
	l = len(delater)
	while h < l:
            # print(instances[delater[h][0]])
            # print(delater[h][1])
            instances[delater[h][0]].remove(delater[h][1])
            
            if not instances[delater[h][0]]:
                del instances[delater[h][0]]
            if not instances:
                print("No more Instances anymore")
                key = ord("q")
            h += 1


	# if the `q` key was pressed, break from the loop and clean the delater list for next frame
	delater = []
	# print(car2.p)
	cv2.imshow("Frame", frame)
	if key == ord("q") or k == kmax:
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
