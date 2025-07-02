import joblib

embeddings, names = joblib.load("models/embeddings.pkl")
print(set(names))
