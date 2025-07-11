import os
import face_recognition
import numpy as np
import joblib

#this is the path to training images
DATA_DIR = "data/train"


# this are Lists to store embeddings and labels
X = []
y = []

# this Loops through each person folder
for person_name in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"Processing: {person_name}")

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        print(f" - {img_path}")

        image = face_recognition.load_image_file(img_path)

        # thi sto find face locations
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"   ⚠️  No face found in {img_name}. Skipping.")
            continue

        # this gets face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            X.append(encoding)
            y.append(person_name)

print(f"✅ Extracted {len(X)} face embeddings.")

# models folder
os.makedirs("../models", exist_ok=True)

#to Save embeddings & labels
joblib.dump({'embeddings': X, 'names': y}, "../models/embeddings.pkl")
print("✅ Saved embeddings to models/embeddings.pkl")
