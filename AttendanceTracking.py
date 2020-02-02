from imutils.video import WebcamVideoStream
import cv2
import numpy as np
from PIL import Image,ImageTk
import PIL
import pytesseract
import cv2
from tkinter import *
import tkinter as tk
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep

cap = WebcamVideoStream(0).start()

#root
root = Tk()
imgWidth = 540
imgHeight = 360
#canvas for image
canvas = Label(root,height = imgHeight,width=imgWidth)
canvas.pack()

lblStudentAttendance = Label(root,text = "Student Attendance")
lblStudentAttendance.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()
lbl = Label(root)
lbl.pack()

def get_encoded_faces():
    encoded = {}
    
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                #face = cv2.imread("faces/"+f)
                face = cv2.resize(face,(135,90))
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

def classify_face(img):
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name.split("-")[0], (left -20, bottom + 15), font, 0.5, (255, 255, 255), 2)

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    new = []
    for i in known_face_names:
        splitName = i.split("-")[0]
        if i in face_names:
            if [splitName,False] not in new:
                new.append([splitName,True])
            else:
                presentIndex = new.index([splitName,False])
                new[presentIndex][1]=True
        else:
            if [splitName,True] not in new and [splitName,False] not in new:
                new.append([splitName,False])
                
    return(new)

def show_frame(count):
    frame = cap.read()
    frame = cv2.resize(frame,(imgWidth,imgHeight))
    students = classify_face(frame)
    if(count == 5):
        for i in range(len(students)):
            color = "red"
            if students[i][1] == True:
                color = 'green'
            nameLabel = tk.Label(root,text = students[i][0],bg=color)
            nameLabel.place(x = 20 + int(i/6)*140, y = imgHeight+30 + (i%6)*30, width=120, height=25)
        count = 0
    #process image
    #frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.configure(image=imgtk)
    canvas.after(1, show_frame,count+1)

show_frame(0)
root.mainloop()


