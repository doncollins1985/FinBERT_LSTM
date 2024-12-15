import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import json


def train_model(input_file, output_model_file, history_file='models/training_history.json'):
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    # Load sequences
    try:
        data = np.load(input_file)
        X, y = data['features'], data['labels']
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Validate data shapes
    assert len(X.shape) == 3, f"Expected X to be 3D, got {X.shape}"
    assert len(y.shape) == 1 or len(
        y.shape) == 2, f"Expected y to be 1D or 2D, got {y.shape}"

    # Split data into train and test sets
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Training samples: {
          X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Normalize features and labels
    feature_scaler = MinMaxScaler()
    label_scaler = MinMaxScaler()

    # Reshape X for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    # Fit scaler on training data and transform
    X_train_scaled = feature_scaler.fit_transform(
        X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(
        X_test_reshaped).reshape(X_test.shape)

    # Fit scaler on training labels and transform
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train_scaled = label_scaler.fit_transform(y_train).flatten()
    y_test_scaled = label_scaler.transform(y_test).flatten()

    print("Data normalization complete.")

    # Define FinBERT-LSTM model
    model = Sequential([
        LSTM(70, activation='tanh', input_shape=(
            X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(30, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(10, activation='tanh'),
        Dense(1)  # Output layer for stock price
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
    print("Model compiled successfully.")
    model.summary()

    # Define callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        'models/best_finbert_lstm_model.keras', monitor='val_loss', save_best_only=True)
    callbacks = [early_stop, checkpoint]

    # Train model
    try:
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_scaled, y_test_scaled),
            callbacks=callbacks
        )
        print("Model training complete.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save training history
    try:
        with open(history_file, 'w') as f:
            json.dump(history.history, f)
        print(f"Training history saved to {history_file}")
    except Exception as e:
        print(f"Error saving training history: {e}")

    # Save the trained model
    try:
        model.save(output_model_file)
        print(f"Model saved to {output_model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Save scalers for future inverse transformations
    try:
        np.savez('models/scalers.npz', feature_scaler=feature_scaler,
                 label_scaler=label_scaler)
        print("Scalers saved to models/scalers.npz")
    except Exception as e:
        print(f"Error saving scalers: {e}")


# Example usage
if __name__ == "__main__":
    train_model("data/sequences.npz", "models/finbert_lstm_model.keras")
