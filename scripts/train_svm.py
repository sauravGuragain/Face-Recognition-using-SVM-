import joblib
from sklearn.svm import SVC

# Load embeddings
X, y = joblib.load("models/embeddings.pkl")

# Train SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# Save the trained model
joblib.dump(clf, "models/svm_model.pkl")
print("Saved SVM model to models/svm_model.pkl")
