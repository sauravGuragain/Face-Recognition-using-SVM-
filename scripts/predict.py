import face_recognition
import joblib
import os
import glob
import cv2

# -------------------------------
# 1️⃣ Load trained SVM model
# -------------------------------
clf = joblib.load("../models/face_recognition_svm.pkl")

# -------------------------------
# 2️⃣ Find all test images
# -------------------------------
test_images = glob.glob("../data/test/*.*")

if not test_images:
    print("❌ No test images found in ../data/test/. Add some images first!")
    exit()

# -------------------------------
# 3️⃣ Make results folder if needed
# -------------------------------
results_folder = "../results"
os.makedirs(results_folder, exist_ok=True)

# -------------------------------
# 4️⃣ Loop through all test images
# -------------------------------
for img_path in test_images:
    print(f"\n📷 Testing: {os.path.basename(img_path)}")

    # Load image
    image = face_recognition.load_image_file(img_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect faces
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes)

    if len(encodings) == 0:
        print("   ⚠️  No face found. Skipping.")
        continue

    for i, (encoding, box) in enumerate(zip(encodings, boxes)):
        pred = clf.predict([encoding])[0]
        prob = clf.predict_proba([encoding])[0].max()

        print(f"   ✅ Prediction: {pred} (Confidence: {prob:.2f})")

        # Draw box & label
        top, right, bottom, left = box
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            rgb_image,
            f"{pred} ({prob:.2f})",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Save result image
    output_path = os.path.join(results_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, rgb_image)
    print(f"   💾 Saved result to {output_path}")
