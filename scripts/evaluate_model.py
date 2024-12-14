import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def evaluate_model(model_file, input_file):
    # Load model and test data
    model = tf.keras.models.load_model(model_file)
    data = np.load(input_file)
    X, y = data['features'], data['labels']
    split = int(0.85 * len(X))
    X_test, y_test = X[split:], y[split:]

    # Predict and evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    print(f"MAE: {mae}, MAPE: {mape}")

    # Plot predictions
    plt.plot(y_test, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    evaluate_model("models/finbert_lstm_model.h5", "data/sequences.npz")

