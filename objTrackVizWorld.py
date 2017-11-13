#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:10:25 2017

"""
import sys
import cv2

def printUsage():
    print("Usage: python3 objTrackVizWorld.py")

def main(args=sys.argv[1:]):
    if len(args) < 1:
        printUsage()
    
if __name__ == "__main__":
    main()