
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("Data/experiments/6runs_parallel_tripPlusTrafficlight/outputDenseP.csv", header= None)

data = pd.read_csv("Data/training_data/outputDenseP.csv", header= None)

plt.figure(figsize=(10, 6)) 
plt.plot(data.iloc[:, 0], marker='o', linestyle='-', color='red', label='Average Total waiting time') 
plt.xlabel('Run number')
plt.ylabel('Average Total waiting time')

plt.legend()

plt.show()
