import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def calculate_pairwise_distances(csv_file):
    data = pd.read_csv(csv_file).to_numpy()
    
    # Calculate pairwise Euclidean distances
    pairwise_distances = cdist(data, data, metric='euclidean')
    
    return pairwise_distances

def plot_distance_heatmap_bw(distance_matrix):
    plt.figure(figsize=(10, 8))
    # Use a grayscale colormap
    plt.imshow(distance_matrix, cmap='gray', aspect='auto', origin='lower')
    plt.colorbar(label='Euclidean Distance')
    
    # Add contour lines for additional distinction
    plt.contour(distance_matrix, colors='black', linewidths=0.5, origin='lower')
    
    plt.title("Pairwise Euclidean Distances Between [TL configs+ State]s")
    plt.xlabel("[TL configs+ State]s")
    plt.ylabel("[TL configs+ State]s")
    plt.grid(visible=False)  # Optional: turn off grid if it adds visual clutter
    plt.show()

if __name__ == "__main__":
    csv_file = "Data/training_data/inputDenseP.csv"
    distance_matrix = calculate_pairwise_distances(csv_file)
    plot_distance_heatmap_bw(distance_matrix)
