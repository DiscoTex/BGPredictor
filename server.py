from flask import Flask, request, jsonify, render_template, send_file
import os  # Import the os module
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv  # Import load_dotenv
import re  # Add this import for input sanitization

# Load environment variables from .env file
load_dotenv()

# Import functions from predict2.py
import predict2

app = Flask(__name__)

# Read environment variables
NIGHTSCOUT_URL = os.environ.get("NIGHTSCOUT_URL")
if not NIGHTSCOUT_URL:
    raise ValueError("NIGHTSCOUT_URL environment variable is not set. Please set it in your .env file.")

MONGODB_URL = os.environ.get("MONGODB_URL")
if not MONGODB_URL:
    print("WARNING: MONGODB_URL environment variable is not set. Data will be fetched from Nightscout instead.")

# Default model name from .env file
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "glucose_predictor")
if not DEFAULT_MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set. Please set it in your .env file.")

# Update helper function to allow letters, digits, underscores, hyphens, and dots in the model path input
def sanitize_model_path(model_path):
    """
    Allow only letters, digits, underscores, hyphens, and dots in the model path.
    """
    if not re.match(r'^[A-Za-z0-9_.\-]+$', model_path):
        raise ValueError("Invalid model path. Only letters, digits, underscores, hyphens, and dots are allowed.")
    return model_path

@app.route('/')
def index():
    # Pass the NIGHTSCOUT_URL and default model name from the environment to the template.
    return render_template('index.html', 
                           nightscout_url=NIGHTSCOUT_URL,
                           model_name=DEFAULT_MODEL_NAME)

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Retrain the model using a large dataset. If MONGODB_URL is provided, 
    5000 entries are fetched from MongoDB; otherwise, 2000 entries are downloaded from Nightscout.
    Uses the model name specified in the UI (form parameter "model") if provided;
    otherwise, falls back to the default from the environment.
    After training, the model is evaluated and an HTML report is returned.
    """
    # Get model path from the form; if not provided, use default.
    model_path = request.form.get("model", DEFAULT_MODEL_NAME)
    try:
        model_path = sanitize_model_path(model_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        if MONGODB_URL:
            output_file, data = predict2.get_mongodb_data(MONGODB_URL, count=5000)
        else:
            print("MONGODB_URL not set, fetching data from Nightscout...")
            output_file, data = predict2.get_nightscout_data(NIGHTSCOUT_URL, count=2000)
    except Exception as e:
        return jsonify({"error": f"Error retrieving data: {str(e)}"}), 500

    try:
        model, scaler, seq_length = predict2.train_glucose_model(data, model_path)
    except Exception as e:
        return jsonify({"error": f"Error retraining model: {str(e)}"}), 500

    try:
        # Evaluate the newly trained model.
        # This function writes the performance report to 'model_performance.html'
        print("Evaluating model performance...")
        predict2.evaluate_model_performance(model, data, scaler, seq_length, report_path='model_performance.html')
    except Exception as e:
        return jsonify({"error": f"Error evaluating model: {str(e)}"}), 500

    # Return the evaluation report HTML to the client.
    return send_file('model_performance.html', mimetype='text/html')

@app.route('/predict', methods=['GET'])
def predict():
    """
    Load an existing model and predict the next 12 glucose values using data fetched 
    from the Nightscout URL specified in the environment variable. Uses the model name 
    specified in the UI (query parameter "model") if provided; otherwise, defaults to 
    the environment value. Also returns the most recent 5 actual glucose values.
    """
    print("Predicting glucose values...")
    url = NIGHTSCOUT_URL
    # Get model path from the query parameters; if not provided, use default.
    model_path = request.args.get("model", DEFAULT_MODEL_NAME)
    try:
        model_path = sanitize_model_path(model_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        output_file, data = predict2.get_nightscout_data(url, count=500)
    except Exception as e:
        return jsonify({"error": f"Error retrieving data: {str(e)}"}), 500

    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found. Retrain the model first."}), 400

    try:
        model = tf.keras.models.load_model(model_path)
        scaler = MinMaxScaler()
        sgv_values = np.array([entry['sgv'] for entry in data]).reshape(-1, 1)
        scaler.fit(sgv_values)

        seq_length = 20  # Assuming sequence length is 20 based on prepare_sequences
        X, _ = predict2.prepare_sequences(data)
        if X.shape[0] == 0:
            return jsonify({"error": "Not enough data to prepare sequences for prediction."}), 400

        X_sgv = scaler.transform(X[:,:,0].reshape(-1, 1)).reshape(X.shape[0], X.shape[1], 1)
        X_day = X[:,:,1].reshape(X.shape[0], X.shape[1], 1) / 6.0
        X_hour = X[:,:,2].reshape(X.shape[0], X.shape[1], 1) / 23.0
        if X.shape[2] > 3:
            X_dom = X[:,:,3].reshape(X.shape[0], X.shape[1], 1)
            X_scaled = np.concatenate([X_sgv, X_day, X_hour, X_dom], axis=2)
        else:
            X_scaled = np.concatenate([X_sgv, X_day, X_hour], axis=2)

        last_sequence = X_scaled[-1:]
        predictions = predict2.predict_glucose_values(model, scaler, seq_length, last_sequence, predict_steps=12)
        predictions = [float(p) for p in predictions]
        recent = [float(entry['sgv']) for entry in data[-5:]]
    except Exception as e:
        print(f"Error in /predict route: {e}", file=sys.stderr)
        return jsonify({"error": f"Error generating predictions: {str(e)}"}), 500

    return jsonify({"predictions": predictions, "recent": recent})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0')