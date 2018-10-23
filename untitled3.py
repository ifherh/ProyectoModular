#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:47:04 2018

@author: Fernanda
"""

import cv2
import glob
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="video")
args = vars(ap.parse_args())


# Playing video from file:
cap = cv2.VideoCapture(args["video"])



try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 10000
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
    
        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        
        # To stop duplicate images
        currentFrame += 1
    
    else: 
        break
    

forexample = "./data/"

eg = cv2.imread(forexample+'frame10000.jpg')
height , width , layers =  eg.shape

print("ok, got that ", height, " ", width, " ", layers)

def makeVideo(imgPath, videodir, videoname, width, height):
    video = cv2.VideoWriter(videodir+videoname,-1,20.0,(width, height))
    for img in sorted(os.listdir(imgPath)):
        if not img.endswith('.jpg'):
            continue
        print(str(imgPath+img))
        shot = cv2.imread(imgPath+img)
        video.write(shot)
    video.release()
    print("one video done")


makeVideo(forexample,forexample, "example.mp4", width, height)

cv2.destroyAllWindows()





















