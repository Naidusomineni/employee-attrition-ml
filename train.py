import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/sample_data.csv")

X = df[["age", "salary", "years"]]
y = df["attrition"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully!")
