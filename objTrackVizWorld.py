#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:10:25 2017

"""
import sys
import cv2
import numpy as np

## used from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')￼
minor_ver = 3

def printUsage():
    print("Usage: python3 objTrackVizWorld.py")

def main(args=sys.argv[1:]):
  
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
    
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
 
    # Read video
    video = cv2.VideoCapture("videos/netwon_2.mp4")
 
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
     
    # print(type(frame))
    # print(frame.shape)
    # getting an initial subset of Areas of interest
    # np_frame = cv2.imread('video', frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_filt = cv2.medianBlur(gray_frame, 5)
    img_new = np.empty_like(img_filt)
    print(img_filt.shape)
    cv2.imshow("imshow", gray_frame)
    # img_th = cv2.adaptiveThreshold(img_filt,img_new,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img_th = cv2.adaptiveThreshold(img_filt, img_new, 255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=3, param1=5) 
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     
    
    print(contours)
    
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()