import pandas as pd
import numpy as np

def create_sequences(input_file, window_size, output_file):
    data = pd.read_csv(input_file)
    features, labels = [], []
    for i in range(len(data) - window_size):
        features.append(data.iloc[i:i + window_size, 1:].values)
        labels.append(data.iloc[i + window_size, 1])  # Predict closing price
    np.savez(output_file, features=np.array(features), labels=np.array(labels))
    print(f"Sequences saved to {output_file}")

# Example usage
if __name__ == "__main__":
    create_sequences("data/merged_data.csv", window_size=10, output_file="data/sequences.npz")

