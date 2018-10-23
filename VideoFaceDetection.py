# import the necessary packages
# import the necessary packages
from imutils.video import FileVideoStream
import imutils
import numpy as np
import argparse
import time
import cv2
# construct the argument parse and parse the arguments
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

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs=FileVideoStream(args["video"]).start()
time.sleep(2.0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter()


# loop over the frames from the video stream
while vs.more():
    
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=480)
    fshape = frame.shape
    fheight = fshape[0]
    fwidth = fshape[1]

    fps = 20.0
    
    # Define the codec and create VideoWriter Object

    out.open('output.mp4',fourcc,fps, (fheight, fwidth),True)


    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
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
        
        #np.random.shuffle(frame[startY:startY+h,startX:startX+w])

	# show the output frame
    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


out.release()
vs.stop()
print("ya acabe")
cv2.destroyAllWindows()

#This method uses deep learning, in particular a Single Shot Detector (SSD) with ResNet base network architecture.
#read
#python VideoFaceDetection.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --video video.mov
