from bargain import generate_observation, generate_observations
from bargain.networks import H5Dataset

import numpy as np
# Generate the training data for the neural networks
generate_observations(filename = "data/train.h5",
                      n_obs = 1000000,
                      n_depots = 3,
                      n_customers = 9,
                      radius = [0.3, 0.4, 0.6])


"""
dataset = H5Dataset(path = "data/instances.h5",
                    features = ['instance', 'coalitions'],
                    target = 'char_function')

seed = 0
np.random.seed(seed)
instance, assignments, coal, char, shap, nucl = generate_observation(n_depots = 3,
                                                                     n_customers = 9,
                                                                     radius = 0.4,
                                                                     show_plots = True)

print(char)
"""