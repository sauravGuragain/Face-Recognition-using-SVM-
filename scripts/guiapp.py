import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import joblib
import os
from PIL import Image, ImageTk
from playsound import playsound

# THIS TO Load trained SVM model
clf = joblib.load("../models/face_recognition_svm.pkl")


# This is the GUI setup using tinker
root = tk.Tk()
root.title("Face Recognition GUI")


root.geometry("900x700")


video_label = tk.Label(root)
video_label.pack()


running = False

# this is for sound it have track name
last_name = None

#  This is for Camera using loop function
def start_camera():
    global running
    running = True
    video_loop()

def stop_camera():
    global running
    running = False

def video_loop():
    global running, last_name

    
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to grab frame.")
            break

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            probs = clf.predict_proba([encoding])[0]
            best_idx = probs.argmax()
            best_prob = probs[best_idx]
            pred = clf.classes_[best_idx]

            # This Draw's box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{pred} ({best_prob:.2f})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # this to play welcome sound if new person and confident
            if best_prob > 0.7 and pred != last_name:
                mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "welcome.mp3")
                mp3_path = mp3_path.replace("\\", "/")  # Fix for playsound Windows issue
                playsound(mp3_path)
                last_name = pred

        # To Convert frame for Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
        root.update_idletasks()
        root.update()

    cap.release()
    video_label.config(image='')

def on_close():
    stop_camera()
    root.destroy()

# for Buttons
btn_start = tk.Button(root, text="Start Camera", command=start_camera, bg="green", fg="white", height=2, width=20)
btn_start.pack(pady=10)

btn_stop = tk.Button(root, text="Stop Camera", command=stop_camera, bg="red", fg="white", height=2, width=20)
btn_stop.pack(pady=10) 
root.protocol("WM_DELETE_WINDOW", on_close)


# this Run GUI

root.mainloop()
