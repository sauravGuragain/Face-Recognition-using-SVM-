import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import joblib
from PIL import Image, ImageTk

# -------------------------------
# 1️⃣ Load trained SVM model
# -------------------------------
clf = joblib.load("../models/face_recognition_svm.pkl")

# -------------------------------
# 2️⃣ GUI setup
# -------------------------------
root = tk.Tk()
root.title("Face Recognition GUI")

# Window size
root.geometry("900x700")

# Video label for display
video_label = tk.Label(root)
video_label.pack()

# Global flag to control camera loop
running = False

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
    global running

    # Open camera
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to grab frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        boxes = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            pred = clf.predict([encoding])[0]
            prob = clf.predict_proba([encoding])[0].max()

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{pred} ({prob:.2f})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Convert frame to ImageTk
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Show in label
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update GUI
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

# -------------------------------
# 5️⃣ Close safely
# -------------------------------
root.protocol("WM_DELETE_WINDOW", on_close)

# -------------------------------
# 6️⃣ Run GUI
# -------------------------------
root.mainloop()
