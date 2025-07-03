import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import joblib
from PIL import Image, ImageTk
from playsound import playsound

# -------------------------------
# 1️⃣ Load trained SVM model
# -------------------------------
clf = joblib.load("../models/face_recognition_svm.pkl")

# -------------------------------
# 2️⃣ GUI setup
# -------------------------------
root = tk.Tk()
root.title("Face Recognition GUI")


root.geometry("900x700")


video_label = tk.Label(root)
video_label.pack()


running = False

# Track last name for sound
last_name = None

# -------------------------------
# 3️⃣ Camera loop function
# -------------------------------
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

            # Draw box and label
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

            # Play welcome sound if new person and confident
            if best_prob > 0.7 and pred != last_name:
                playsound("welcome.mp3")  # Ensure you have this in same folder
                last_name = pred

        # Convert frame for Tkinter
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

# -------------------------------
# 4️⃣ Buttons
# -------------------------------
btn_start = tk.Button(root, text="Start Camera", command=start_camera, bg="green", fg="white", height=2, width=20)
btn_start.pack(pady=10)

btn_stop = tk.Button(root, text="Stop Camera", command=stop_camera, bg="red", fg="white", height=2, width=20)
btn_stop.pack(pady=10) 
root.protocol("WM_DELETE_WINDOW", on_close)

# -------------------------------
# 5️⃣ Run GUI
# -------------------------------
root.mainloop()
