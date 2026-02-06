from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        age = int(request.form["age"])
        salary = int(request.form["salary"])
        years = int(request.form["years"])

        result = model.predict(np.array([[age, salary, years]]))[0]
        prediction = "Will Leave" if result == 1 else "Will Stay"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
