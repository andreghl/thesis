from bargain.networks import train, GainNN, tune
from bargain import generate_observations
from tests import test_gain_net
import numpy as np

"""
print("Generating observations...")
generate_observations(filename = "data/instances.h5",
                      obs = 100000,
                      vehicles = 3,
                      customers = 9,
                      radius = 1.0)

print("Generating observations for hyperparameter tuning...")
generate_observations(filename = "data/tune.h5",
                      obs = 100000,
                      vehicles = 3,
                      customers = 9,
                      radius = 1.0,
                      seed = 2026)
"""
parameters = {
    "learning_rate": (1e-3, 1e-1),
    "weight_decay": (0.0, 1e-3),
    "batch_size": (32, 180)}

print("Tuning Gain network...")
score, params = tune(model = GainNN(),
                     parameters = parameters,
                     n_models = 10,
                     data_path = "data/tune.h5",
                     features = ["instance", "coalitions"],
                     target = "gain",
                     label = "GainNN",
                     tune_epochs = 15)
print(f"Selected parameters: {params} with score {score}")

print("Training Gain network...")
train(**params,
      model = GainNN(),
      n_epochs = 30,
      data_path = "data/instances.h5",
      features = ["instance", "coalitions"],
      target = "gain",
      label = "GainNN",
      verbose = 0)