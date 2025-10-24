import pandas as pd
import matplotlib.pyplot as plt

# csv_file = 'D:/intersection_lane_vehiclesPLow.csv'
csv_file = 'Data/training_data/intersection_lane_vehiclesP.csv'
data = pd.read_csv(csv_file,on_bad_lines='skip')
# data = pd.read_csv(csv_file, on_bad_lines='skip')

# Convert 'Total Vehicle Count' to numeric, forcing errors to NaN
data['Total Vehicle Count'] = pd.to_numeric(data['Total Vehicle Count'], errors='coerce')

# Drop rows with NaN 
data = data.dropna(subset=['Total Vehicle Count'])

# Find junctions with the highest total vehicle count
top_2_junctions = data.groupby('Junction ID')['Total Vehicle Count'].sum().nlargest(6).index

# Filter data for the top 3 junctions
filtered_data = data[data['Junction ID'].isin(top_2_junctions)]

fig, axs = plt.subplots(1, 6, figsize=(25, 6))

for i, junction in enumerate(top_2_junctions):
    junction_data = filtered_data[filtered_data['Junction ID'] == junction]
    
    # Count the number of states for each unique total vehicle count
    states_count = junction_data['Total Vehicle Count'].value_counts().sort_index()
    # plot barChart
    #axs[i].bar(states_count.index, states_count.values, alpha=0.7)
    # plot Histogram
    axs[i].hist(junction_data['Total Vehicle Count'], bins=10, alpha=0.7, edgecolor='black')

    
    axs[i].set_xlabel('Number of Cars', fontsize = 14)
    # axs[i].set_ylabel('Number of States')
    axs[i].set_title(f' IntersectionID: {junction}')
    axs[i].set_xticks(states_count.index)  # Show all x-tick labels
    axs[i].set_ylim(0 , 23000)
      # Show every 10th tick from the sorted index
    xtick_positions = states_count.index[::10]
    axs[i].set_xticks(xtick_positions)
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)
    # axs[i].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
