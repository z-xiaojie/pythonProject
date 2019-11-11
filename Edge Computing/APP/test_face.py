import cv2
import numpy as np
import os
import face_recognition
import time


def load_known_face():
    # Load some sample pictures and learn how to recognize them.
    samples = ["./known_face/lin-manuel-miranda.png", "./known_face/alex-lacamoire.png", "./known_face/xiaojie_zhang.png"]
    known_faces = []
    for item in samples:
        img = face_recognition.load_image_file(item)
        known_faces.append(face_recognition.face_encodings(img)[0])
    return known_faces


def one_step(frame, known_faces):
    start = time.time()
    rgb_frame = frame[:, :, ::-1]
    # rgb_frame = cv2.resize(frame, (416, 416))
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=2)
    face_names = []
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        name = None
        if match[0]:
            name = "Lin-Manuel Miranda"
        elif match[1]:
            name = "Alex Lacamoire"
        elif match[2]:
            name = "Xiaojie Zhang"
        face_names.append(name)
    print("finished in", time.time() - start, face_names)


known_faces = load_known_face()
frame = cv2.imread("./known_face/xiaojie_zhang.png")
for i in range(1):
    start = time.time()
    rgb_frame = frame[:, :, ::-1]
    # rgb_frame = cv2.resize(frame, (416, 416))
    face_locations = face_recognition.face_locations(rgb_frame)
    new_img = []
    new_location = []
    for face in face_locations:
        new_img.append(rgb_frame[face[0]:face[2], face[3]:face[1]])
        new_location.append([[0, face[1] - face[3], face[2] - face[0], 0]])
        # cv2.imshow('image', new_img)
        # cv2.waitKey(0)
    print("step 1 finished in", time.time() - start)

    total_size = 0
    for img in new_img:
        total_size += img.nbytes

    print("step 1 data", total_size/1024, "frame size", rgb_frame.nbytes/1024)

    start = time.time()
    face_names = []
    for j in range(len(new_img)):
        face_encodings = face_recognition.face_encodings(new_img[j], new_location[j], num_jitters=2)
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
            name = None
            if match[0]:
                name = "Lin-Manuel Miranda"
            elif match[1]:
                name = "Alex Lacamoire"
            elif match[2]:
                name = "Xiaojie Zhang"
            face_names.append(name)
    print("step 2 finished in", time.time() - start, face_names)

