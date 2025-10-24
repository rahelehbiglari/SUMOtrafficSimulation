import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def calculate_pairwise_distances(csv_file):
    data = pd.read_csv(csv_file).to_numpy()
    
    # Multiply the first 640 columns by 0.1
    # data[:, 641:] = data[:, 641:] * 100

    
    # Calculate pairwise Euclidean distances
    pairwise_distances = cdist(data, data, metric='euclidean')
    # pairwise_distances = cdist(data, data, metric='sqeuclidean')

    
    return pairwise_distances

def plot_distance_heatmap(distance_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='coolwarm_r', aspect='auto', origin='lower')  # Set origin to 'lower' to compare from lower
    plt.colorbar(label='squared Euclidean Distance')
    plt.title("Pairwise squared Euclidean Distances Between [TL configs+ State]s")
    plt.xlabel("[TL configs+ State]s")
    plt.ylabel("[TL configs+ State]s")
    plt.show()

if __name__ == "__main__":
    csv_file = "Data/training_data/inputDenseP.csv"
    distance_matrix = calculate_pairwise_distances(csv_file)
    plot_distance_heatmap(distance_matrix)
