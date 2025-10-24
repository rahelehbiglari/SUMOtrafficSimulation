# Define the model (a simple linear regression model)
import sys
import os

# Get the directory of the current script
script_parent_dir = os.path.dirname(os.path.abspath(os.getcwd()))

# Get the parent directory of the script
# Add the parent directory to the Python path if it's not already there
if script_parent_dir not in sys.path:
    sys.path.insert(0, script_parent_dir)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Read the first CSV file
df1 = pd.read_csv("Data/training_data/inputDenseP.csv", header=None)

# Read the second CSV file
df2 = pd.read_csv("Data/training_data/outputDenseP.csv", header=None)

# Concatenate the two DataFrames horizontally (column-wise)
combined_df = pd.concat([df1, df2], axis=1)

combined_df.to_csv('Data/training_data/training_file.csv', header=False, index=False)

data = pd.read_csv("Data/training_data/training_file.csv", header=None)

nn_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the model (a simple linear regression model)
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(8096, 1)  # 8069 input feature and one output

    def forward(self, x):
        return self.linear(x)

# Define the loss function (Mean Squared Error)
loss_fn = nn.MSELoss()

model = LinearRegression()

model

# Just a simple sample just to get an idea of the shape of the data accepted by the neural network
m = nn.Linear(8096, 1)
input = torch.randn(100, 8096) # Create 100 samples with the right shape
print(input.size())
output = m(input)
print(output.size())

# Separate training datafrom test data
# Calculate the number of rows to drop (30% of the total rows)
rows_to_drop = int(0.3 * len(nn_data))

# Use iloc to select all rows except the last 30%
training_data = nn_data.iloc[:-rows_to_drop]


test_data = nn_data.iloc[-rows_to_drop:]


assert(len(training_data) + len(test_data) == len(nn_data))

# Transform pandas data into torch tensors
def convert_to_torch_inputs_outputs(df):
  # Convert the NumPy array to a PyTorch tensor
  X = torch.tensor(df.iloc[:,:8096].values, dtype=torch.float32)

  # Same for the desired predictions
  Y = torch.tensor(df.iloc[:,8096:8097].values, dtype=torch.float32)

  assert(X.size()[0] == Y.size()[0])
  return X, Y

X_train, Y_train = convert_to_torch_inputs_outputs(training_data)
print(X_train)
print(Y_train)

learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5000  # Choose the number of training epochs

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero out gradients
    outputs = model(X_train)  # Forward pass
    loss = loss_fn(outputs, Y_train)  # Compute the loss
    loss.backward()  # Backpropagation
    
    optimizer.step()  # Update the model's parameters

    # Print the loss for monitoring training progress
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# get the full data predictions back into a table
X_full, _ = convert_to_torch_inputs_outputs(nn_data)

nn_data[8097] = model(X_full).detach().numpy()
nn_data
print(nn_data)

# and plot
plt.plot(nn_data.iloc[:, 8096], label='waitingTime')
plt.plot(nn_data.iloc[:, 8097], label='Predicted waitingTime')
plt.ylabel('Averate total waiting time(s)')
plt.xlabel('Samples')
plt.legend()

# Show the plots
plt.show()