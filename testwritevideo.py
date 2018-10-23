#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:47:04 2018

@author: Fernanda
"""

import cv2
import shutil
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", required=True,
	help="video")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Playing video from file:
cap = cv2.VideoCapture(args["video"])



try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 100000
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        
        #frame = imutils.resize(frame, width=300)
    
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        		(300, 300), (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
        
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue
        
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
        
            # draw the bounding box of the face along with the associated
            # probability
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),0)
            
            weight=endX-startX
            height=endY-startY
            
            
            subframe=frame[startY:startY+height,startX:startX+weight]
            for x in range(0,startX+weight,30):
                yy = x - 30 if x - 30 > 30 else x + 30
                
                for y in range(0,yy+height,30):
                    xx = y - 30 if y - 30 > 30 else y + 30
                    #col = [r, g, b] = frame[x, y]
                    
    
                    # COLORES
                    
                    cv2.rectangle(subframe, (xx,x), (y,yy),(np.random.randint(120,140),np.random.randint(120,135),np.random.randint(130,170)),cv2.FILLED)
                
                    
            frame[startY:startY+height,startX:startX+weight]=subframe
            
        
        
    
        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        
        # To stop duplicate images
        currentFrame += 1
    
    else: 
        break
    
    cv2.imshow("Frame", frame)
    #out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    
    	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    

pathtoread = "./data/"
pathtowrite = "./"
videoname=np.random.randint(0,10000)

eg = cv2.imread(pathtoread+'frame100000.jpg')
height , width , layers =  eg.shape

print("ok, got that ", height, " ", width, " ", layers)

def makeVideo(imgPath, videodir, videoname, width, height):
    video = cv2.VideoWriter(videodir+videoname,-1,18.0,(width, height))
    for img in sorted(os.listdir(imgPath)):
        if not img.endswith('.jpg'):
            continue
        print(str(imgPath+img))
        shot = cv2.imread(imgPath+img)
        video.write(shot)
    video.release()
    print("one video done")


makeVideo(pathtoread,pathtowrite, "example2"+str(videoname)+".mp4", width, height)

shutil.rmtree("./data/")
cv2.destroyAllWindows()





















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:25:01 2018

@author: Fernanda
"""

