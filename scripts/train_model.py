import joblib
import os
from sklearn.svm import SVC

# 1️⃣ Load embeddings and labels (saved as tuple)
embeddings, labels = joblib.load('../models/embeddings.pkl')

print(f"[INFO] Loaded {len(embeddings)} embeddings with {len(set(labels))} unique classes.")

# 2️⃣ Train SVM classifier
print("[INFO] Training SVM classifier...")
model = SVC(C=1.0, kernel="linear", probability=True)
model.fit(embeddings, labels)

# 3️⃣ Save trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/face_recognition_svm.pkl")

print("✅ SVM model trained and saved to models/face_recognition_svm.pkl")
