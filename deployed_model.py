# Import libraries
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
# Load the model
model = pickle.load(open('rf_model.pkl','rb'))
with open('transformations.json') as f:
     transformations = json.load(f)
     mean, std = transformations['n_bike']['mean'], transformations['n_bike']['std']


@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    inputs = np.array(data['inputs'])

    prediction = (model.predict(inputs) * std + mean) ** 2

    return jsonify(list(prediction))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
