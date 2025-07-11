import joblib

data = joblib.load("../models/embeddings.pkl")  # note the ../ to go one folder up
embeddings = data["embeddings"]
names = data["names"]

print(set(names))
