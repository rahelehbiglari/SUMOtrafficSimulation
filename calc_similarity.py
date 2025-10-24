import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_distances(csv_file):
    data = pd.read_csv(csv_file)

    # Use a zero vector as the reference vector
    reference_vector = np.zeros(data.shape[1])
    
    distances = []
    for index, row in data.iterrows():
        row_vector = row.to_numpy()
        distance = np.linalg.norm(reference_vector - row_vector)
        distances.append(distance)

    return distances

def plot_distances(distances):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, marker='o', linestyle='', color='b', label='Similarity to zero')

    # Annotate each point with its row index
    for i, distance in enumerate(distances):
        plt.annotate(str(i), (i, distance), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='darkred')

    plt.title("Similarity of Each [TL configs+ State] to Zero")
    plt.xlabel("[TL configs+ State]s (Row Index)")
    plt.ylabel("Euclidean Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_file = "Data/training_data/inputDenseP.csv"
    distances = calculate_distances(csv_file)
    plot_distances(distances)
