import logging
from d2l import torch as d2l

import Trainer
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from syne_tune import StoppingCriterion, Tuner, Reporter
from syne_tune.backend.python_backend import python_backend
from syne_tune.config_space import loguniform, randint
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import HyperbandScheduler


def objective(config):
    """Hyper parameter tuning script ended up unused"""
    model = Trainer.MLP(lr=config["learning_rate"], num_hidden=config["num_hidden"], num_outputs=config['num_outputs'])
    trainer = d2l.HPOTrainer()

    input_data = pd.read_csv("Data/training_data/Legacy-training-data-files/input2.csv", header=None)
    output_data = pd.read_csv("Data/training_data/Legacy-training-data-files/output2.csv", header=None)

    X = input_data.values
    y = output_data.values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    data = Trainer.Data(X, y, batch_size=30)

