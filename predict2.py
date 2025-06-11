import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# GPU Detection and Configuration
def configure_gpu():
    """Configure GPU settings. Returns True if GPU is enabled; else falls back to CPU."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth to avoid consuming all GPU memory
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"GPU enabled: Found {len(physical_devices)} GPU device(s)")
            return True
        except RuntimeError as e:
            print(f"Warning: Error configuring GPU - {e}")
            print("Falling back to CPU")
            return False
    else:
        print("No GPU devices found - using CPU")
        return False

# Configure GPU or fallback to CPU
using_gpu = configure_gpu()

# Import other dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError  # For model loss calculation
import subprocess
import json
import sys
from datetime import datetime

def get_nightscout_data(url, count=1000):
    """
    Query Nightscout via HTTP (using curl) to download the most recent entries.
    
    Parameters:
        url (str): Nightscout base URL.
        count (int): Number of entries to retrieve.
    
    Returns:
        tuple: (output_file, data_as_dict) where output_file is the JSON filename
               and data_as_dict is a list of entry dictionaries.
               
    Raises:
        ValueError: If the URL does not start with 'http://' or 'https://'.
        Exception: If the downloaded file is missing or empty, or data format is invalid.
    """
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    
    output_file = 'nightscout_data.json'
    curl_cmd = [
        'curl',
        '-X', 'GET',
        f"{url}/api/v1/entries.json?count={count}",
        '-H', 'accept: application/json',
        '-o', output_file
    ]
    try:
        result = subprocess.run(curl_cmd, check=True, capture_output=True, text=True)
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise Exception("Failed to download data or empty response received")
        with open(output_file, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Invalid data format - expected JSON array")
        
        df = pd.DataFrame(data)
        date_field = 'dateString' if 'dateString' in df.columns else 'date'
        if date_field in df.columns:
            df['datetime'] = pd.to_datetime(df[date_field])
            df['day_of_week'] = df['datetime'].dt.day_name()
        else:
            print("Warning: No date field found in the data")
          
        df = df.sort_values(by='datetime')
        csv_file = 'nightscout_data.csv'
        df.to_csv(csv_file, index=False)
        print(f"Successfully downloaded {len(data)} entries from Nightscout to {output_file} and {csv_file}")
        return output_file, df.to_dict('records')
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing curl command: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def get_mongodb_data(mongo_url, count=1000):
    """
    Query MongoDB directly using the provided connection URL.
    
    Parameters:
        mongo_url (str): The MongoDB connection URI.
        count (int): Number of entries to retrieve.
    
    Returns:
        tuple: (output_file, data_as_dict) where output_file is the JSON filename
               and data_as_dict is a list of entry dictionaries.
               
    Raises:
        Exception: If querying MongoDB fails.
    """
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_url)  # mongo_url is the MongoDB URI
        db = client['nightscout']        # Assuming the Nightscout DB name is "nightscout"
        collection = db['entries']       # Using the "entries" collection
        cursor = collection.find().sort("date", -1).limit(count)
        data = list(cursor)
    except Exception as e:
        print(f"Error querying MongoDB: {e}")
        sys.exit(1)
    
    output_file = 'nightscout_data_mongodb.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, default=str)
    
    df = pd.DataFrame(data)
    date_field = 'dateString' if 'dateString' in df.columns else 'date'
    if date_field in df.columns:
        df['datetime'] = pd.to_datetime(df[date_field])
        df['day_of_week'] = df['datetime'].dt.day_name()
    else:
        print("Warning: No date field found in MongoDB data")
    
    df = df.sort_values(by='datetime')
    csv_file = 'nightscout_data_mongodb.csv'
    df.to_csv(csv_file, index=False)
    print(f"Retrieved {len(data)} entries from MongoDB and saved to {output_file} and {csv_file}")
    return output_file, df.to_dict('records')

def prepare_sequences(data):
    """
    Prepare sequences for LSTM model training and prediction.
    
    Each sequence consists of 20 consecutive entries and includes three features:
      - 'sgv': Glucose value.
      - Day of week encoded as an integer (0 for Monday to 6 for Sunday).
      - Hour of the day extracted from 'dateString'.
    
    Parameters:
        data (list): List of dictionaries containing glucose entry data.
    
    Returns:
        tuple: (X, y) where X is a NumPy array of input sequences and y is a NumPy array of targets.
    """
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    
    sequences = []
    targets = []
    
    for i in range(len(data) - 20):
        seq = data[i:i+20]
        target = data[i+20]
        
        if 'sgv' in seq[0] and 'day_of_week' in seq[0]:
            seq_values = [
                [
                    entry['sgv'],
                    day_mapping[entry['day_of_week']],
                    datetime.fromisoformat(entry['dateString'].replace('Z', '+00:00')).hour
                ] 
                for entry in seq
            ]
            sequences.append(seq_values)
            targets.append(target['sgv'])
    
    X = np.array(sequences, dtype=float)  # Ensure data is float type
    y = np.array(targets)
    return X, y

def build_glucose_predictor(seq_length, num_features):
    # Instead of passing input_shape to the LSTM layer, we start with an Input layer.
    model = Sequential([
        Input(shape=(seq_length, num_features)),
        LSTM(64, return_sequences=False),
        Dense(1)  # Adjust as needed for your output
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_glucose_model(data, model_path='glucose_predictor'):
    """
    Train an LSTM model on prepared glucose data.
    
    The function prepares sequences, scales features, trains the model,
    and saves it in SavedModel format.
    
    Parameters:
        data (list): List of dictionaries containing glucose data.
        model_path (str): Path/directory to save the trained model.
    
    Returns:
        tuple: (model, scaler, sequence_length)
    """
    try:
        # Prepare sequences and targets from data
        X, y = prepare_sequences(data)
        
        # Scale the 'sgv' feature and normalize day and hour features
        scaler = MinMaxScaler()
        X_sgv = scaler.fit_transform(X[:,:,0].reshape(-1, 1)).reshape(X.shape[0], X.shape[1], 1)
        X_day = X[:,:,1].reshape(X.shape[0], X.shape[1], 1) / 6.0  # Normalize day of week (0-6)
        X_hour = X[:,:,2].reshape(X.shape[0], X.shape[1], 1) / 23.0  # Normalize hour (0-23)
        X_scaled = np.concatenate([X_sgv, X_day, X_hour], axis=2)
        y_scaled = scaler.transform(y.reshape(-1, 1))
        
        # Create and train the LSTM model
        print("Training new model...")
        model = build_glucose_predictor(20, 3)  # Sequence length is 20, 3 features per timestep
        print(f"Using {'GPU' if using_gpu else 'CPU'} for training")
        
        history = model.fit(
            X_scaled, 
            y_scaled, 
            epochs=50, 
            batch_size=32 if using_gpu else 16,  # Adjust batch size based on device
            validation_split=0.1, 
            verbose=1
        )
        
        # Save the trained model
        model.save(model_path, save_format='tf')
        print(f"Model saved to {model_path}")
            
        return model, scaler, X.shape[1]
    
    except Exception as e:
        print(f"Error in training model: {e}")
        raise

def predict_glucose_values(model, scaler, seq_length, last_sequence, predict_steps=12):
    """
    Iteratively predict the next glucose values using the trained model.
    
    For each prediction, the model output is inverse-transformed and appended to 
    the sequence to predict subsequent values.
    
    Parameters:
        model (tf.keras.Model): Trained LSTM model.
        scaler (MinMaxScaler): Scaler used for feature scaling.
        seq_length (int): Length of input sequence.
        last_sequence (np.ndarray): The most recent sequence for prediction.
        predict_steps (int): Number of future glucose values to predict.
    
    Returns:
        list: List of predicted glucose values.
    """
    try:
        predictions = []
        
        for i in range(predict_steps):
            pred = model.predict(last_sequence, verbose=0)
            predictions.append(scaler.inverse_transform(pred)[0][0])
            
            # Update sequence for next prediction by removing the oldest record and adding the new prediction.
            new_seq = last_sequence[0][1:].copy()
            new_day = (last_sequence[0][-1][1] + 1/288) % 1  # Update day feature for 5-min interval advancement
            new_hour = (last_sequence[0][-1][2] + 5/60) % 1     # Update hour feature for 5-min interval advancement
            new_val = np.array([[pred[0][0], new_day, new_hour]])
            last_sequence = np.concatenate([new_seq, new_val]).reshape(1, seq_length, 3)
        
        return predictions
    
    finally:
        tf.keras.backend.clear_session()

def evaluate_model_performance(model, data, scaler, seq_length, report_path='model_performance.html'):
    """
    Evaluate the model by comparing predictions against actual values on a sampled subset.
    
    Generates an HTML report containing scatter plots for each prediction time step and 
    a separate plot for the Mean Absolute Error (MAE) over time.
    
    Parameters:
        model (tf.keras.Model): Trained LSTM model.
        data (list): List of dictionaries containing glucose entries.
        scaler (MinMaxScaler): Scaler used for input normalization.
        seq_length (int): The sequence length used for prediction.
        report_path (str): Path to save the performance report HTML.
    """
    # Ensure sufficient data is available for evaluation
    min_required = seq_length + 12
    if len(data) < min_required:
        print(f"\nSkipping performance evaluation - insufficient data")
        print(f"Need at least {min_required} entries, but only have {len(data)}")
        return
    
    X, _ = prepare_sequences(data)
    if X.shape[0] == 0:
        print("\nSkipping performance evaluation - no valid sequences generated")
        print("This may happen if the data doesn't contain required fields")
        return
        
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Limit the total number of evaluation samples to a maximum (e.g. 50)
    total_evaluations = len(data) - seq_length - 12
    max_samples = 50
    sample_interval = max(1, total_evaluations // max_samples)
    
    results = []
    for i in range(0, total_evaluations, sample_interval):
        sequence = data[i:i+seq_length]
        actual_values = [entry['sgv'] for entry in data[i+seq_length:i+seq_length+12]]
        
        # Prepare sequence for prediction; append a dummy record to allow processing
        X_temp, _ = prepare_sequences([*sequence, {'sgv': 0, 'day_of_week': 'Monday',
                                                    'dateString': '2000-01-01T00:00:00Z'}])
        if X_temp.shape[0] > 1:
            X_temp = X_temp[:-1]
        
        # Scale and normalize each feature
        X_sgv = scaler.transform(X_temp[:,:,0].reshape(-1, 1)).reshape(X_temp.shape[0], X_temp.shape[1], 1)
        X_day = X_temp[:,:,1].reshape(X_temp.shape[0], X_temp.shape[1], 1) / 6.0
        X_hour = X_temp[:,:,2].reshape(X_temp.shape[0], X_temp.shape[1], 1) / 23.0
        X_scaled = np.concatenate([X_sgv, X_day, X_hour], axis=2)
        last_sequence = X_scaled[-1:]
        
        # Predict the next 12 time steps
        predictions = predict_glucose_values(model, scaler, seq_length, last_sequence)
        
        results.append({
            'time': datetime.fromisoformat(sequence[-1]['dateString'].replace('Z', '+00:00')),
            'predictions': predictions,
            'actual': actual_values
        })
    
    # Create subplots for scatter plots of each prediction time step (t+5 to t+60 minutes)
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[f't+{i*5} minutes' for i in range(1, 13)],
        vertical_spacing=0.12
    )
    
    metrics = []
    for step in range(12):
        actual = [r['actual'][step] for r in results]
        pred = [r['predictions'][step] for r in results]
        mse = np.mean((np.array(actual) - np.array(pred)) ** 2)
        mae = np.mean(np.abs(np.array(actual) - np.array(pred)))
        metrics.append({'step': step+1, 'mse': mse, 'mae': mae})
        
        row = (step // 3) + 1
        col = (step % 3) + 1
        
        # Scatter plot for predicted vs. actual values
        fig.add_trace(
            go.Scatter(
                x=actual, y=pred,
                mode='markers',
                name=f't+{(step+1)*5}min',
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Diagonal reference line for perfect prediction
        min_val = min(min(actual), min(pred))
        max_val = max(max(actual), max(pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Axis labeling for clarity
        fig.update_xaxes(title_text="Actual Glucose (mg/dL)", row=row, col=col)
        fig.update_yaxes(title_text="Predicted Glucose (mg/dL)", row=row, col=col)
    
    fig.update_layout(
        title='Prediction Performance by Time Step (Sampled Evaluation)',
        height=1000,
        width=1200,
        showlegend=False
    )
    
    # Plot for Mean Absolute Error (MAE) across prediction time steps
    time_steps = [(step+1)*5 for step in range(12)]
    mae_values = [m['mae'] for m in metrics]
    
    mae_fig = go.Figure()
    mae_fig.add_trace(go.Scatter(
        x=time_steps, y=mae_values,
        mode='lines+markers',
        name='MAE'
    ))
    mae_fig.update_layout(
        title='Mean Absolute Error by Prediction Time Step',
        xaxis_title='Time Step (minutes)',
        yaxis_title='MAE (mg/dL)',
        height=500,
        width=600
    )
    
    # Generate HTML report embedding both figures and the metrics table
    with open(report_path, 'w') as f:
        f.write('''
        <html>
        <head>
            <title>Glucose Prediction Model Performance</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
        <h1>Glucose Prediction Model Performance Report</h1>
        ''')
        
        # Metrics table
        f.write('''
        <h2>Prediction Metrics</h2>
        <table>
            <tr><th>Time Step</th><th>MSE</th><th>MAE</th></tr>
        ''')
        for m in metrics:
            f.write(f'<tr><td>t+{m["step"]*5}min</td><td>{m["mse"]:.2f}</td><td>{m["mae"]:.2f}</td></tr>')
        f.write('</table>')
        
        # Embed the scatter subplots figure
        f.write('<h2>Predicted vs Actual Glucose</h2>')
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        # Embed the MAE line plot
        f.write('<h2>Mean Absolute Error (MAE) by Prediction Time Step</h2>')
        f.write(mae_fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        f.write('</body></html>')
    
    print(f"Performance report saved to {report_path}")

if __name__ == "__main__":
    # Parse command line arguments for configuration
    import argparse
    
    parser = argparse.ArgumentParser(description='Glucose prediction from Nightscout data')
    parser.add_argument('--url', default=os.getenv('NIGHTSCOUT_URL'),
                        help='Nightscout URL (or set NIGHTSCOUT_URL env variable)')
    parser.add_argument('--mongo', default=os.getenv('MONGODB_URL'),
                        help='MongoDB connection URL')
    parser.add_argument('--model', default='glucose_predictor',  # Without file extension
                        help='Path to existing model directory (optional)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force training a new model even if one exists')
    
    args = parser.parse_args()
    
    if not args.url:
        print("Please provide Nightscout URL as argument or set NIGHTSCOUT_URL environment variable")
        sys.exit(1)
        
    # Choose entry count based on retraining flag
    if args.retrain:
        entry_count = 26000  # Use a larger count for retraining (approximately 3 months of data)
        output_file, data = get_nightscout_data(args.mongo, count=entry_count)
    else:
        entry_count = 600  # Smaller count for quick predictions
        output_file, data = get_nightscout_data(args.url, count=entry_count)
    
    # Train a new model or load an existing model based on the provided arguments
    model_path = args.model
    
    if not os.path.exists(model_path) or args.retrain:
        model, scaler, seq_length = train_glucose_model(data, model_path)    
    else:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        scaler = MinMaxScaler()
        scaler.fit(np.array([entry['sgv'] for entry in data]).reshape(-1, 1))
        seq_length = 20 
        
    # Prepare the last sequence for prediction
    X, _ = prepare_sequences(data)
    X_sgv = scaler.transform(X[:,:,0].reshape(-1, 1)).reshape(X.shape[0], X.shape[1], 1)
    X_day = X[:,:,1].reshape(X.shape[0], X.shape[1], 1) / 6.0
    X_hour = X[:,:,2].reshape(X.shape[0], X.shape[1], 1) / 23.0
    X_scaled = np.concatenate([X_sgv, X_day, X_hour], axis=2)
    last_sequence = X_scaled[-1:]
    
    # Get historical data for display (last 5 entries)
    historical_data = [(entry['dateString'], int(entry['sgv'])) for entry in data[-5:]]
    
    # Make predictions for the next 12 time steps
    predictions = predict_glucose_values(model, scaler, seq_length, last_sequence, predict_steps=12)
    
    # Convert the last timestamp to local timezone
    last_time = datetime.fromisoformat(data[-1]['dateString'].replace('Z', '+00:00'))
    local_tz = datetime.now().astimezone().tzinfo
    last_time = last_time.astimezone(local_tz)
    
    print("\nLast 5 glucose values:")
    for i, (time_str, value) in enumerate(historical_data):
        time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        local_time = time.astimezone(local_tz)
        print(f"{local_time.strftime('%I:%M:%S %p')} ({value:3d} mg/dL)")
    
    print("\nPredicted next 12 glucose values:")
    for i, pred in enumerate(predictions, 1):
        future_time = last_time + pd.Timedelta(minutes=i*5)
        print(f"{future_time.strftime('%I:%M:%S %p')} ({int(pred):3d} mg/dL)")
    
    if args.retrain:
        print("\nModel retrained successfully!")
    print("\nGenerating model performance report...")
    evaluate_model_performance(model, data, scaler, seq_length)
    print("\nDone!")
    sys.exit(0)
