import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def calculate_pairwise_distances(csv_file1, csv_file2):
    data1 = pd.read_csv(csv_file1).to_numpy()
    data2 = pd.read_csv(csv_file2).to_numpy()
    
    # Calculate pairwise Euclidean distances between the two datasets
    pairwise_distances = cdist(data1, data2, metric='euclidean')
    # pairwise_distances = cdist(data1, data2, metric='sqeuclidean')  # Uncomment for squared Euclidean distances
    
    return pairwise_distances

def plot_distance_heatmap(distance_matrix, labels1, labels2):
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='coolwarm_r', aspect='auto', origin='lower')  # Set origin to 'lower' to compare from lower
    plt.colorbar(label='Euclidean Distance')
    plt.title("Pairwise Euclidean Distances Between Two Datasets")
    plt.xlabel("Dataset 2")
    plt.ylabel("Dataset 1")
    plt.xticks(ticks=np.arange(len(labels2)), labels=labels2, rotation=90)
    plt.yticks(ticks=np.arange(len(labels1)), labels=labels1)
    plt.show()

if __name__ == "__main__":
    csv_file1 = "Data/training_data/inputDenseP.csv"  # Dataset 1
    csv_file2 = "Data/training_data/inputDenseP.csv"  # Dataset 2
    
    # Load datasets for labels (optional, if you want to display indices/names)
    labels1 = pd.read_csv(csv_file1).index
    labels2 = pd.read_csv(csv_file2).index
    
    distance_matrix = calculate_pairwise_distances(csv_file1, csv_file2)
    plot_distance_heatmap(distance_matrix, labels1, labels2)
