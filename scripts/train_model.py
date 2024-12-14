import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def train_model(input_file, output_model_file):
    # Load sequences
    data = np.load(input_file)
    X, y = data['features'], data['labels']

    # Split data into train and test sets
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Define FinBERT-LSTM model
    model = Sequential([
        LSTM(70, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        LSTM(30, activation='tanh', return_sequences=True),
        LSTM(10, activation='tanh'),
        Dense(1)  # Output layer for stock price
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(output_model_file)
    print(f"Model saved to {output_model_file}")

# Example usage
if __name__ == "__main__":
    train_model("data/sequences.npz", "models/finbert_lstm_model.h5")

