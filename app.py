# Making a Flask API for model
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__)

# Load the pickle model

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]  # Convert the integer values and the values in float types
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)