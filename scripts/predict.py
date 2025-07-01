import face_recognition
import joblib

# Load model
clf = joblib.load("models/svm_model.pkl")

# Load test image
image = face_recognition.load_image_file("data/test/person1/test.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

for encoding, location in zip(face_encodings, face_locations):
    result = clf.predict([encoding])
    print("Predicted label:", result)
