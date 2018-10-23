#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:06:09 2018

@author: Fernanda
"""

# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
 
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # draw the bounding box of the face along with the associated
        # probability
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),0)
        
        weight=endX-startX
        height=endY-startY
        
        
        subframe=image[startY:startY+height,startX:startX+weight]
        for x in range(0,startX+weight,30):
            yy = x - 30 if x - 30 > 30 else x + 30
            
            for y in range(0,yy+height,30):
                xx = y - 30 if y - 30 > 30 else y + 30
                #col = [r, g, b] = frame[x, y]
                

                # COLORES
                
                cv2.rectangle(subframe, (xx,x), (y,yy),(np.random.randint(120,140),np.random.randint(120,135),np.random.randint(130,170)),cv2.FILLED)
            
                
        image[startY:startY+height,startX:startX+weight]=subframe
 
cv2.imwrite('outputimage.png',image)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)