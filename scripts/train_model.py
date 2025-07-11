import joblib
import os
from sklearn.svm import SVC

# ✅ Load embeddings and labels (dict, not tuple!)
data = joblib.load("../models/embeddings.pkl")
embeddings = data["embeddings"]
labels = data["names"]

print(f"[INFO] Loaded {len(embeddings)} embeddings with {len(set(labels))} unique classes.")

# ✅ Train SVM classifier
print("[INFO] Training SVM classifier...")
model = SVC(C=1.0, kernel="linear", probability=True)
model.fit(embeddings, labels)

# ✅ Save trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/face_recognition_svm.pkl")

print("✅ SVM model trained and saved to models/face_recognition_svm.pkl")
