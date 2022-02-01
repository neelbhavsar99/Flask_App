from flask import Flask, render_template, Response
from imutils.video import FPS
import cv2
import imutils
import argparse
import numpy as np
import os

application = Flask(__name__)
# camera = cv2.VideoCapture(0)

# prototxtPath = os.path.dirname(os.path.abspath(
#     'application.py')) + '/server/Caffe/SSD_MobileNet_prototxt.txt'
# modelPath = os.path.dirname(os.path.abspath(
#     'application.py')) + '/server/Caffe/SSD_MobileNet.caffemodel'

# def generate_frame():
    
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-c", "--confidence", type=float, default=0.7)
#     args = vars(ap.parse_args())
#     print("args: " + str(args))

#     # Initialize Objects and corresponding colors which the model can detect
#     labels = [
#         "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
#         "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
#         "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
#     ]
#     colors = np.random.uniform(0, 255, size=(len(labels), 3))

#     # Loading Caffe Model
#     print('[Status] Loading Model...')
#     # nn = cv2.dnn.readNetFromCaffe(args[prototxtPath], args[modelPath])
#     nn = cv2.dnn.readNetFromCaffe(prototxtPath, modelPath)

#     # Initialize Video Stream
#     print('[Status] Starting Video Stream...')

#     #vs = VideoStream(src=0).start()
#     # time.sleep(2.0)
#     fps = FPS().start()

#     # Loop Video Stream
#     while True:
#         success, frame = camera.read()  # read camera frame continuosly

#         # Resize Frame to 400 pixels
#         #frame = vs.read()
#         frame = imutils.resize(frame, width=400)
#         (h, w) = frame.shape[:2]

#         # Converting Frame to Blob
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
#                                      (300, 300), 127.5)

#         # Passing Blob through network to detect and predict
#         nn.setInput(blob)
#         detections = nn.forward()

#         # Loop over the detections
#         for i in np.arange(0, detections.shape[2]):

#             # Extracting the confidence of predictions
#             confidence = detections[0, 0, i, 2]

#             # Filtering out weak predictions
#             if confidence > args["confidence"]:

#                 # Extracting the index of the labels from the detection
#                 # Computing the (x,y) - coordinates of the bounding box
#                 idx = int(detections[0, 0, i, 1])

#                 # Extracting bounding box coordinates
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 print("StartX: " + str(startX) + "\tEndX: " + str(endX))
#                 print("StartY: " + str(startY) + "\tEndY: " + str(endY))

#                 # Drawing the prediction and bounding box
#                 label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
#                 cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx],
#                               2)

#                 y = startY - 15 if startY - 15 > 15 else startY + 15
#                 cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5, colors[idx], 2)

#                 # encode frame into multiple pics
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()  # computer vision requires this
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         #cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('q'):
#             break

#         fps.update()

#     fps.stop()


# @application.route('/video')
# def video():
#     # Response will call some function
#     return Response(generate_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route('/',methods=['GET','POST'])
def index():
    #return "Flask app is running"
    return render_template('index.html')

if __name__ == "__main__":
    application.run()