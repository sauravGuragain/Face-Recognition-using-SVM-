import cv2
import face_recognition
import joblib

#loads train modela
clf = joblib.load("../models/face_recognition_svm.pkl")

#this opens webcam
video_capture = cv2.VideoCapture(0)  # 0 = default camera

if not video_capture.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🎥 Starting webcam. Press 'q' to quit.")

#this to loop throgh frames
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Convert BGR (OpenCV) -> RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, boxes)

    # Loop through each face found
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

    # Show the frame
    cv2.imshow("Live Face Recognition", frame)

    # Break loop if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#cleanup
video_capture.release()
cv2.destroyAllWindows()
print("👋 Webcam stopped. Bye!")
