from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import webbrowser

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/heart_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = [
            data["age"],
            data["sex"],
            data["cp"],
            data["trestbps"],
            data["chol"],
            data["fbs"],
            data["restecg"],
            data["thalach"],
            data["exang"],
            data["oldpeak"],
            data["slope"],
            data["ca"],
            data["thal"]
        ]

        input_data = np.array(features).reshape(1, -1)
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        risk = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability * 100, 2),
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)