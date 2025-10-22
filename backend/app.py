import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Load the Model and Vectorizer ---
# Construct the full path to the model and vectorizer files
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
model_path = os.path.join(model_dir, 'kmeans_model.joblib')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')

# Load the model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer files not found. Make sure you have run the train_model.py script.")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500

    # Get the project description from the request
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Invalid input: "description" not found in request body'}), 400

    description = data['description']

    # Vectorize the new description
    X_new = vectorizer.transform([description])

    # Predict the cluster
    prediction = model.predict(X_new)

    # Return the prediction
    cluster_id = int(prediction[0])
    return jsonify({'cluster': cluster_id})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
