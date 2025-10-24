import matplotlib.pyplot as plt
import pandas as pd
import torch
from d2l import torch as d2l
from torch import nn


class MLP(d2l.Module):

    def __init__(self, num_outputs, num_hidden, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.LazyLinear(num_hidden), nn.ReLU(),
                                 nn.LazyLinear(num_hidden), nn.ReLU(),
                                 nn.LazyLinear(num_hidden), nn.ReLU(),
                                 nn.LazyLinear(num_hidden), nn.ReLU(),
                                 nn.LazyLinear(num_hidden), nn.ReLU(),
                                 nn.LazyLinear(num_outputs))

    def loss(self, y_hat, y, averaged=True):
        fn = nn.L1Loss()
        return fn(y_hat, y)


class Data(d2l.DataModule):
    def __init__(self, X, y, num_train=4500, num_val=1500, # num_train = 4500, num_val=1500 for low density #num_train=5185, num_val=1800 for high density
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.X = X
        self.y = y

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


if __name__ == "__main__":
    """Script to train a neural network using input_data and output_data"""

    input_data = pd.read_csv("Data/training_data/inputLowDense.csv", header=None)
    output_data = pd.read_csv("Data/training_data/outputLowDense.csv", header=None)

    X = input_data.values
    y = output_data.values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    data = Data(X, y, batch_size=30)
    # data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    model = MLP(num_outputs=1, num_hidden=50, lr=0.0003)
    trainer = d2l.Trainer(max_epochs=1000)
    trainer.fit(model, data)

    if 'val_loss' in model.board.data:
        print(float(model.board.data['val_loss'][-1].y))
    else:
        print("'val_loss' key not found in model.board.data")


    print(float(model.board.data['val_loss'][-1].y))
    model.eval()
    with torch.no_grad():
        print(model(X[20]))
        print(y[20])
    torch.save(model.state_dict(), 'mlpDense.params')

    plt.show()
