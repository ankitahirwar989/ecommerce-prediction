
from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load trained model
model = xgb.XGBClassifier()
model.load_model("model.json")

FEATURE_COUNT = 84


@app.route("/")
def home():
    return render_template("index.html", feature_count=FEATURE_COUNT)


@app.route("/predict", methods=["POST"])
def predict():

    features = []

    for i in range(FEATURE_COUNT):
        value = request.form.get(f"feature{i}")
        features.append(float(value))

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Customer will PURCHASE"
    else:
        result = "Customer will NOT PURCHASE"

    return render_template("index.html",
                           prediction_text=result,
                           feature_count=FEATURE_COUNT)


if __name__ == "__main__":
    app.run(debug=True)
