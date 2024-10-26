from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the ML model API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Ensure the input data is in the correct format
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    # Extract features from the request
    features = data['features']
    
    # Convert features to a numpy array and reshape for the model
    features = np.array(features).reshape(1, -1)
    
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
