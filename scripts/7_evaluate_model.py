import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os


def evaluate_model(model_file, input_file, scaler_file):
    """
    Evaluates a trained LSTM model on test data.

    Parameters:
    - model_file (str): Path to the saved Keras model.
    - input_file (str): Path to the .npz file containing test data.
    - scaler_file (str): Path to the saved scalers (.npz file).

    Outputs:
    - Prints MAE and MAPE metrics.
    - Displays a plot comparing actual and predicted stock prices.
    """

    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return

    # Check if input data file exists
    if not os.path.exists(input_file):
        print(f"Input data file not found: {input_file}")
        return

    # Check if scaler file exists
    if not os.path.exists(scaler_file):
        print(f"Scaler file not found: {scaler_file}")
        return

    # Load scalers
    try:
        scaler_data = np.load(scaler_file, allow_pickle=True)
        feature_scaler = scaler_data['feature_scaler'].item()
        label_scaler = scaler_data['label_scaler'].item()
        print("Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # Load the model without compiling to avoid issues with custom objects
    try:
        model = tf.keras.models.load_model(model_file, compile=False)
        print("Model loaded successfully without compiling.")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Recompile the model with the same settings as during training
    try:
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Error compiling model: {e}")
        return

    # Load test data
    try:
        data = np.load(input_file)
        X, y = data['features'], data['labels']
        print("Test data loaded successfully.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Validate data shapes
    if len(X.shape) != 3:
        print(f"Expected X to be 3D, got {X.shape}")
        return
    if len(y.shape) != 1 and len(y.shape) != 2:
        print(f"Expected y to be 1D or 2D, got {y.shape}")
        return

    # Split the data (assuming data was split during training)
    split = int(0.85 * len(X))
    X_test, y_test = X[split:], y[split:]
    print(f"Number of test samples: {X_test.shape[0]}")

    # Scale the test features
    try:
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = feature_scaler.transform(
            X_test_reshaped).reshape(X_test.shape)
        print("Test features scaled successfully.")
    except Exception as e:
        print(f"Error scaling test features: {e}")
        return

    # Scale the test labels
    try:
        y_test = y_test.reshape(-1, 1)
        y_test_scaled = label_scaler.transform(y_test).flatten()
        print("Test labels scaled successfully.")
    except Exception as e:
        print(f"Error scaling test labels: {e}")
        return

    # Make predictions
    try:
        predictions_scaled = model.predict(X_test_scaled)
        print("Predictions made successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Ensure predictions are flattened
    predictions_scaled = predictions_scaled.flatten()

    # Inverse transform predictions and true labels to original scale
    try:
        predictions = label_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)).flatten()
        y_test_original = label_scaler.inverse_transform(
            y_test_scaled.reshape(-1, 1)).flatten()
        print("Inverse transformations applied successfully.")
    except Exception as e:
        print(f"Error during inverse transformation: {e}")
        return

    # Calculate evaluation metrics on original scale
    try:
        mae = mean_absolute_error(y_test_original, predictions)
        mape = mean_absolute_percentage_error(y_test_original, predictions)
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    # Plot predictions vs actual values
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label='Actual Prices', color='blue')
        plt.plot(predictions, label='Predicted Prices', color='red')
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Sample')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
        return


# Example usage
if __name__ == "__main__":
    evaluate_model(
        # Path to the best model saved during training
        model_file="models/checkpoints/best_finbert_lstm_model.keras",
        input_file="data/sequences.npz",                 # Path to the data file
        scaler_file="models/scalers.npz"                 # Path to the scalers
    )
