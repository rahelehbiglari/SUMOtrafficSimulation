import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

def calculate_pairwise_distances(data, pca_components=100):
    # Perform PCA to reduce dimensions for computational efficiency
    pca = PCA(n_components=pca_components)
    reduced_data = pca.fit_transform(data)
    
    # Calculate pairwise distances in reduced space
    distances = pdist(reduced_data, metric='euclidean')
    distance_matrix = squareform(distances)
    return distance_matrix

def plot_distance_heatmap(distance_matrix, title="Pairwise Distances Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='coolwarm_r', aspect='auto', origin='lower')
    plt.colorbar(label='Euclidean Distance')
    plt.title(title)
    plt.xlabel("Row Index")
    plt.ylabel("Row Index")
    plt.show()

def remove_close_rows_based_on_heatmap(data, distance_matrix, distance_threshold):
    # Identify rows to remove based on the threshold
    rows_to_remove = set()
    num_samples = distance_matrix.shape[0]
    
    for i in range(num_samples):
        if i in rows_to_remove:
            continue
        for j in range(i + 1, num_samples):
            if j not in rows_to_remove and distance_matrix[i, j] < distance_threshold:
                rows_to_remove.add(j)
    
    # Keep rows that are not marked for removal
    rows_to_keep = [i for i in range(num_samples) if i not in rows_to_remove]
    filtered_data = data.iloc[rows_to_keep]
    return filtered_data, len(rows_to_remove)

if __name__ == "__main__":
    # Load data
    df1LD = pd.read_csv("Data/training_data/inputDenseP.csv", header=None)
    df2LD = pd.read_csv("Data/training_data/outputDenseP.csv", header=None)
    combined_dfLD = pd.concat([df1LD, df2LD], axis=1)
    combined_dfLD.to_csv('Data/training_data/training_fileHD.csv', header=False, index=False)
    csv_file = "Data/training_data/training_fileHD.csv"
    distance_threshold = 50  # You can adjust this based on the heatmap visualization

    # df1LD = pd.read_csv("Data/training_data/inputLowDenseP.csv", header=None)
    # df2LD = pd.read_csv("Data/training_data/outputLowDenseP.csv", header=None)
    # combined_dfLD = pd.concat([df1LD, df2LD], axis=1)
    # combined_dfLD.to_csv('Data/training_data/training_fileLD.csv', header=False, index=False)
    # csv_file = "Data/training_data/training_fileLD.csv"
    # distance_threshold = 50  # You can adjust this based on the heatmap visualization
    
    # Step 1: Load the dataset
    print("Loading data...")
    data = pd.read_csv(csv_file)
    original_row_count = data.shape[0]
    
    # Step 2: Calculate pairwise distances
    print("Calculating pairwise distances...")
    distance_matrix = calculate_pairwise_distances(data, pca_components=100)
    
    # Step 3: Plot initial heatmap
    print("Plotting initial heatmap...")
    plot_distance_heatmap(distance_matrix, title="Initial Pairwise Distances Heatmap")
    
    # Step 4: Remove close rows based on threshold
    print(f"Removing rows with distances < {distance_threshold}...")
    filtered_data, rows_removed_count = remove_close_rows_based_on_heatmap(data, distance_matrix, distance_threshold)
    
    # Step 5: Save the filtered dataset
    # output_file = "Data/training_data/filtered_training_fileHD"
    output_file = "Data/training_data/filtered_training_fileLD.csv"
    filtered_data.to_csv(output_file, index=False)
    
    # Step 6: Report results
    filtered_row_count = filtered_data.shape[0]
    print(f"Filtered data saved to {output_file}.")
    print(f"Original number of rows: {original_row_count}")
    print(f"Number of rows removed: {rows_removed_count}")
    print(f"Number of rows remaining: {filtered_row_count}")
    
    # Step 7: Plot heatmap for filtered data
    print("Calculating pairwise distances for filtered data...")
    filtered_distance_matrix = calculate_pairwise_distances(filtered_data, pca_components=100)
    
    print("Plotting heatmap for filtered data...")
    plot_distance_heatmap(filtered_distance_matrix, title="Filtered Pairwise Distances Heatmap")
