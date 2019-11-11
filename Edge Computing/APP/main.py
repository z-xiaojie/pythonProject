from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import face_recognition
import time


dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/APP"


def create_net(threshold=0.6):
    return TFNet({
           "imgdir": dir + "/sample_frame",
           "model": dir + "/cfg/tiny-yolo-voc.cfg",
           "load": dir + "/bin/tiny-yolo-voc.weights",
           "threshold": float(threshold),
           "batch": 1,
           "json": True
    })


def load_known_face():
    # Load some sample pictures and learn how to recognize them.
    samples = ["./known_face/lin-manuel-miranda.png", "./known_face/alex-lacamoire.png", "./known_face/xiaojie_zhang.png"]
    known_faces = []
    for item in samples:
        img = face_recognition.load_image_file(item)
        known_faces.append(face_recognition.face_encodings(img)[0])
    return known_faces


def check_face(frame, known_faces):
    start = time.time()
    # frame = cv2.resize(frame, (416, 416))
    rgb_frame = frame[:, :, ::-1]
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


def one_image(tfnet, filename):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (360, 640))

    input_movie = cv2.VideoCapture("./sample_video/short_hamilton_clip.mp4")  # VID_20190831_124923.mp4
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    known_faces = load_known_face()
    index = 0
    times = []
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        # Quit when the input video file ends
        if not ret:
            break
        # frame_name = "f_"+str(index)+".jpg"
        # cv2.imwrite("./sample_face/"+frame_name, frame)
        # frame = cv2.imread("./sample_face/"+frame_name)
        # frame = cv2.resize(frame, (416, 416))
        # (h, w) = frame.shape[:2]
        # M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
        # frame = cv2.warpAffine(frame, M, (h, w))

        # cv2.imshow('image', frame)
        # cv2.waitKey(0)
        # start = time.time()
        # result = tfnet.predict(None, frame, only_person=True)
        # print(result["time"])

        # check_face(frame, known_faces)

        """
        for i in range(len(result["person_img"])):
            for person in result["person_img"][i]:
                rgb_frame = person
                # cv2.imshow('image', rgb_frame)
                # cv2.waitKey(0)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
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
                print(index, ">>>>>>>>>>>>>>>", face_locations, face_names)
                # Label the results
                for (top, right, bottom, left), name in zip(result["location"][i], face_names):
                    if not name:
                        continue
                    # Draw a box around the face
                    # frame = cv2.resize(frame, (416, 416))
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        index += 1
        # print(result["result"])
        # times.append(result["time"])
        """
        output_movie.write(frame)

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()

    # print("avg time is", np.average(times))
    return index


model = create_net(threshold=0.3)
start = time.time()
index = one_image(model, None)
print("number of frame", index, time.time() - start)



