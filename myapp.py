from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model and preprocessing objects
model = pickle.load(open("fishmodel.pkl", "rb"))
scaler = pickle.load(open("fishscaler.pkl", "rb"))
label_encoder = pickle.load(open("fishlabel_encoder.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    data = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform([data])
    
    # Predict fish species
    prediction = model.predict(scaled_data)
    species = label_encoder.inverse_transform(prediction)[0]

    return render_template("index.html", prediction_text=f"Predicted Fish Species: {species}")

if __name__ == "__main__":
    app.run(debug=True)
