# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:02:24 2017

@author: nautilus
"""

import sys
import cv2
import numpy as np


def main(args=sys.argv[1:]):
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
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_filt = cv2.medianBlur(gray_frame, 5)
    img_new = np.empty_like(img_filt)
    print(img_filt.shape)
    cv2.imshow("img_filt", img_filt)
    cv2.waitKey(0)   
    # img_th = cv2.adaptiveThreshold(img_filt,img_new,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # img_th = cv2.adaptiveThreshold(img_filt, img_new, 255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=3, param1=5) 
    # contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     
    img_th = cv2.adaptiveThreshold(img_filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    cv2.imshow("img_th", img_th)
    cv2.waitKey(0)   
    # ret,thresh = cv2.threshold(img_filt,127,255,0)
    edges = cv2.Canny(img_th, 30, 200)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)      
    image, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # finding top 10 larges contours:
    topTen = min(len(contours), 30)
    print(topTen)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:topTen]
    
    frame_overlayed = frame.copy()
    for c in contours:
        # approximate the contour
        # peri = cv2.arcLength(c, True)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame_overlayed,(x,y),(x+w,y+h),(0,255,0),2)
        # poly = cv2.approxPolyDP(c, 0.02 * peri, True)
        # cv2.drawContours(frame_overlayed, [poly], -1, (0, 255, 0), 3)


    cv2.imshow("overlayed with rectangles", frame_overlayed)
    cv2.waitKey(0)
    
    print(contours)
    
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()