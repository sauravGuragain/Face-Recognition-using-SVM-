import joblib
from sklearn.svm import SVC

#This Load embeddings
X, y = joblib.load("models/embeddings.pkl")

# This for Train SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# This is to Save the trained model
joblib.dump(clf, "models/svm_model.pkl")
print("Saved SVM model to models/svm_model.pkl")
