from bargain import generate_observation, generate_observations
from bargain.networks import H5Dataset
from tests import test_gain_net
import numpy as np



generate_observations(filename ="data/single_instances.h5",
                      obs = 100000,
                      vehicles = 3,
                      customers = 9,
                      radius = [0.3, 0.4, 0.6])
"""
dataset = H5Dataset(path = "data/instances.h5",
                    features = ['instance', 'coalitions'],
                    target = 'gain')

output, gain = test_gain_net(path = "data/models/GainNN-2026-07-06_19-59-22.pth",
                                seed = 2026); print(output, gain, np.linalg.norm(output - gain))

output, gain = test_gain_net(path = "data/models/GainNN-2026-07-06_21-34-27.pth",
                                seed = 2026); print(output, gain, np.linalg.norm(output - gain))
"""