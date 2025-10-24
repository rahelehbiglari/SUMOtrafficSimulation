import multiprocessing as mp
import random
import pickle
from typing import Tuple, List
import itertools
import math

# Define the computation you want to perform
def compute(x: int, y: int) -> Tuple[int, int, int]:
    # For example, let us compute the largest common factor of x and y
    result = 0
    for ix in range(0, x):
        for iy in range(0, y):
            result += math.gcd(ix,iy)
    return (x, y, result)

# Function that takes a pair of parameters and performs the computation
def worker(params: Tuple[int, int]) -> Tuple[int, int, int]:
    x, y = params
    return compute(x, y)

# Main function to run the computation in parallel
def run_computations(x_samples: List[int], y_samples: List[int], num_workers: int, output_file: str):
    # Create a list of all parameter combinations (Cartesian product of x_samples and y_samples)
    param_combinations = list(itertools.product(x_samples, y_samples))

    # Initialize a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Run the computations in parallel using the worker function
        results = pool.map(worker, param_combinations)

    # Store the results in a pickle file for later analysis
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results stored in {output_file}")

# Example usage
if __name__ == "__main__":
    num_samples = 1000  # Number of samples for each parameter
    num_workers = 8    # Number of parallel workers to use
    output_file = "computation_results.pkl"

    # Generate num_samples random samples for each parameter from a uniform distribution
    x_samples = [round(random.uniform(0, 100)) for _ in range(num_samples)]
    y_samples = [round(random.uniform(0, 100)) for _ in range(num_samples)]

    # Run the computations
    run_computations(x_samples, y_samples, num_workers, output_file)
