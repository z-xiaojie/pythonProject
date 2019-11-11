from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import face_recognition
import time

dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/APP"

def create_net(threshold=0.7):
    return TFNet({
           "imgdir": dir + "/sample_img",
           "model": dir + "/cfg/tiny-yolo-voc.cfg",
           "load": dir + "/bin/tiny-yolo-voc.weights",
           "threshold": float(threshold),
           "batch": 1,
           "json": True
    })

model = create_net(threshold=0.5)

for i in range(1):
    start = time.time()
    result = model.predict("p3_dance.png", None, only_person=True)
    for i in range(len(result["person_img"])):
        for person in result["person_img"][i]:
            rgb_frame = person
            cv2.imshow('image', rgb_frame)
            cv2.waitKey(0)
    print("number of frame", time.time() - start)
    cv2.imshow('image', result["img"][0])
    cv2.waitKey(0)
