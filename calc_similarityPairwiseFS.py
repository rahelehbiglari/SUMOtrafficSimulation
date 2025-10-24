# adding feature scaling to similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Scale the features using standardization or normalization
def scale_features(data, method='standardize'):
    if method == 'standardize':
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    elif method == 'normalize':
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        raise ValueError("Invalid scaling method. Choose 'standardize' or 'normalize'.")
    
    return data_scaled

# Calculate pairwise similarity based on inverse Euclidean distance
def calculate_pairwise_similarity(data):
    # Compute Euclidean distance
    distances = cdist(data, data, metric='euclidean')
    
    # Convert distances to similarity (inverse of distance)
    similarity =  distances
    return similarity

# Plot the similarity matrix as a heatmap
def plot_similarity_heatmap(similarity_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='warmcool', aspect='auto', origin='lower')  # Using 'coolwarm' for contrast
    plt.colorbar(label='Distance')
    plt.title("Pairwise Distance Matrix (Euclidean Distance with feature scaling)")
    plt.xlabel("Data Sample")
    plt.ylabel("Data Sample")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load your dataset (ensure the path is correct)
    data = pd.read_csv("Data/training_data/inputDenseP.csv").to_numpy()

    # Scale the data (standardize or normalize)
    data_scaled_standardized = scale_features(data, method='standardize')
    data_scaled_normalized = scale_features(data, method='normalize')

    # Calculate pairwise similarity for standardized data
    similarity_matrix_standardized = calculate_pairwise_similarity(data_scaled_standardized)

    # Plot similarity matrix for standardized data
    plot_similarity_heatmap(similarity_matrix_standardized)

    # Calculate pairwise similarity for normalized data
    similarity_matrix_normalized = calculate_pairwise_similarity(data_scaled_normalized)

    # Plot similarity matrix for normalized data
    plot_similarity_heatmap(similarity_matrix_normalized)
