import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def predict_next_day(merged_data_file, model_file, scaler_file, window_size=10):
    # Check if files exist
    if not os.path.exists(merged_data_file):
        print(f"Data file not found: {merged_data_file}")
        return
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return
    
    if not os.path.exists(scaler_file):
        print(f"Scaler file not found: {scaler_file}")
        return

    # Load the merged data
    data = pd.read_csv(merged_data_file)
    # Ensure data is sorted by date if not already
    data = data.sort_values(by='Date')

    # Extract features (all columns except 'Date')
    # Assuming columns: [Date, Close, Positive, Negative, Neutral]
    feature_cols = data.columns[1:]  # Exclude 'Date'
    all_values = data[feature_cols].values

    # Ensure we have at least `window_size` days of data
    if len(all_values) < window_size:
        print(f"Not enough data for {window_size}-day window.")
        return
    
    # Get the last `window_size` data points
    recent_values = all_values[-window_size:]
    
    # Load scalers
    scaler_data = np.load(scaler_file, allow_pickle=True)
    feature_scaler = scaler_data['feature_scaler'].item()
    label_scaler = scaler_data['label_scaler'].item()

    # The model was trained on scaled data
    X_input_reshaped = recent_values.reshape(-1, recent_values.shape[-1])
    X_input_scaled = feature_scaler.transform(X_input_reshaped).reshape(1, window_size, -1)

    # Load the model
    model = tf.keras.models.load_model(model_file, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # Predict the next day's price (scaled)
    prediction_scaled = model.predict(X_input_scaled)
    # Inverse transform the prediction to the original scale
    prediction = label_scaler.inverse_transform(prediction_scaled).flatten()[0]

    # Determine the next date after the last date in the dataset
    last_date_str = data['Date'].iloc[-1]
    last_date = pd.to_datetime(last_date_str)
    next_date = last_date + pd.Timedelta(days=1)
    next_date_str = next_date.strftime("%Y-%m-%d")

    # Print the predicted date and price
    print(f"Predicted stock price for {next_date_str}: {prediction}")

if __name__ == "__main__":
    # Set parameters
    MERGED_DATA_FILE = "data/merged_data.csv"
    MODEL_FILE = "models/finbert_lstm_model.keras"
    SCALER_FILE = "models/scalers.npz"
    WINDOW_SIZE = 10  # Adjust if you used a different window size during training

    predict_next_day(MERGED_DATA_FILE, MODEL_FILE, SCALER_FILE, WINDOW_SIZE)

