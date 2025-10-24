import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Load data
df1LD = pd.read_csv("Data/training_data/inputLowDenseP.csv", header=None)
df2LD = pd.read_csv("Data/training_data/outputLowDenseP.csv", header=None)
combined_dfLD = pd.concat([df1LD, df2LD], axis=1)
combined_dfLD.to_csv('Data/training_data/training_fileLD.csv', header=False, index=False)

# df1HD = pd.read_csv("Data/training_data/inputDenseP.csv", header=None)
# df2HD = pd.read_csv("Data/training_data/outputDenseP.csv", header=None)
# combined_dfHD = pd.concat([df1HD, df2HD], axis=1)
# combined_dfHD.to_csv('Data/training_data/training_fileHD.csv', header=False, index=False)

# df1 = pd.read_csv("Data/training_data/training_fileLD.csv", header=None)
# df2 = pd.read_csv("Data/training_data/training_fileHD.csv", header=None)
# combined_df = pd.concat([df1, df2])
# combined_df.to_csv('Data/training_data/training_file.csv', header=False, index=False)

# data = pd.read_csv("Data/training_data/filtered_training_fileHD.csv", header=None)
# data = pd.read_csv("Data/training_data/training_fileHD.csv", header=None)
data = pd.read_csv("Data/training_data/training_fileLD.csv", header=None)
# data = pd.read_csv("Data/training_data/filtered_training_fileLD.csv", header=None)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.iloc[:, :8096])
Y_values = data.iloc[:, 8096:8097].values

data_scaled = pd.DataFrame(X_scaled)
data_scaled[8096] = Y_values  # Append the target column back


nn_data = data_scaled.sample(frac=1, random_state=42).reset_index(drop=True)

# Define simple model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(8096, 1)

    def forward(self, x):
        return self.linear(x)

# Define a deeper model
class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(8096, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

# Separate training and test data
rows_to_drop = int(0.3 * len(nn_data))
training_data = nn_data.iloc[:-rows_to_drop]
test_data = nn_data.iloc[-rows_to_drop:]

# Transform pandas data into torch tensors and move to device
def convert_to_torch_inputs_outputs(df):
    X = torch.tensor(df.iloc[:, :8096].values, dtype=torch.float32).to(device)
    Y = torch.tensor(df.iloc[:, 8096:8097].values, dtype=torch.float32).to(device)
    return X, Y

X_train, Y_train = convert_to_torch_inputs_outputs(training_data)
X_test, Y_test = convert_to_torch_inputs_outputs(test_data)

# Initialize models, move to device, define loss and optimizers
model_simple = LinearRegression().to(device)
model_deep = DeepNeuralNetwork().to(device)
loss_fn = nn.MSELoss()
learning_rate = 0.01
optimizer_simple = optim.Adam(model_simple.parameters(), lr=learning_rate)
optimizer_deep = optim.Adam(model_deep.parameters(), lr=learning_rate)

# Training function
def train_model(model, optimizer, X_train, Y_train, model_label, num_epochs=500):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'[{model_label}]: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

# Train and save both models
model_simple = train_model(model_simple, optimizer_simple, X_train, Y_train, 'simple')
# torch.save(model_simple.state_dict(), "simple_model_HighLowInputDiffTL.pth")
model_deep = train_model(model_deep, optimizer_deep, X_train, Y_train, 'deep')
# torch.save(model_deep.state_dict(), "deep_model_High_DiffTL_retrain.pth")
# torch.save(model_deep.state_dict(), "deep_model_High_DiffTL.pth")
# torch.save(model_deep.state_dict(), "deep_model_Low_DiffTL.pth")
torch.save(model_deep.state_dict(), "deep_model_Low_DiffTL_retrain.pth")
# Print MSE loss during training



# Test and compare function
# def test_and_compare(models, X_test, Y_test):
#     results = {}
#     for name, model in models.items():
#         model.eval()
#         with torch.no_grad():
#             predictions = model(X_test)
#             mse_loss = loss_fn(predictions, Y_test).item()
#             results[name] = (predictions.cpu(), mse_loss)  # Move predictions to CPU for plotting
#     return results

# Test and compare function with manual MSE
def test_and_compare(models, X_test, Y_test):
    results = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            
            # Built-in MSE
            mse_builtin = loss_fn(predictions, Y_test).item()
            
            # Manual MSE calculation
            squared_diff = (predictions - Y_test) ** 2
            mse_manual = torch.mean(squared_diff).item()
            
            # Print both for verification
            print(f"[{name}] Built-in MSE: {mse_builtin:.6f} | Manual MSE: {mse_manual:.6f}")
            
            # Optional: Check they are close
            assert abs(mse_builtin - mse_manual) < 1e-6, f"MSE mismatch in {name} model!"

            # Store results using manual MSE
            results[name] = (predictions.cpu(), mse_manual)
    
    return results


# Compare both models
models = {"simple": model_simple, "deep": model_deep}
results = test_and_compare(models, X_test, Y_test)


# Plotting results
for name, (predictions, mse_loss) in results.items():
    plt.figure()
    plt.plot(Y_test.cpu().numpy(), label='Actual Waiting Time')  # Move Y_test to CPU for plotting
    plt.plot(predictions.numpy(), label=f'{name} Predicted (MSE: {mse_loss:.4f})')
    plt.title(f'Comparison of {name}')
    plt.ylabel('Average Total Waiting Time(s)')
    plt.xlabel('Samples')
    plt.legend()
    plt.show()
