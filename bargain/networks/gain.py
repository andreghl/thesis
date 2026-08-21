import torch.nn as nn
import numpy as np
import torch


def transform_instances(instances: np.ndarray | torch.Tensor,
                        n_coalitions: int,
                        device: torch.device):
    """"""
    batches, rows, cols = instances.shape
    instances = instances[:, :, 1:] if cols == 5 else instances
    _, _, cols = instances.shape

    if isinstance(instances, np.ndarray):
        instances = torch.from_numpy(instances).to(device)
    else:
        instances = instances.to(device)

    instances = instances.unsqueeze(1).repeat(1, n_coalitions, 1, 1)
    # the following should flatten by row
    return instances.reshape(batches * n_coalitions, rows * cols)

def transform_coalitions(coalitions : np.ndarray | torch.Tensor,
                         device: torch.device):
    """"""
    batches, rows, cols = coalitions.shape

    if isinstance(coalitions, np.ndarray):
        coalitions = torch.from_numpy(coalitions).to(device)
    else:
        coalitions = coalitions.to(device)

    return coalitions.reshape(batches * rows, cols)

def transform_values(values: np.ndarray | torch.Tensor,
                     device: torch.device):
    """"""
    batches, cols = values.shape

    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values).to(device)
    else:
        values = values.to(device)

    return values.reshape(batches * cols, 1)

def transform(instances: np.ndarray | torch.Tensor,
              coalitions: np.ndarray | torch.Tensor,
              values: np.ndarray | torch.Tensor,
              device: torch.device):
    """Transform the dataset into the relevant format for the neural network.

    Args:
        instances: A numpy array of the instance matrices.
        coalitions: A numpy array of the coalition vectors.
        values: a numpy array containing the value of a coalition.
        device: A device to run the neural network on.

    Returns:
        Two torch tensor of the instances and coalitions matrices.
    """

    _, n_coalitions, _ = coalitions.shape

    instances = transform_instances(instances,
                                    n_coalitions,
                                    device)

    coalitions = transform_coalitions(coalitions,
                                      device)

    values = transform_values(values, device)

    return instances, coalitions, values

class GainNN(nn.Module):
    """Approximate the value of collaboration of gain of all coalitions
    in a VRP instance with 3 players."""

    def __init__(self,
                 instance_size: int = 40,
                 n_vehicles: int = 3):
        super().__init__()

        self.extractors = nn.ModuleDict({
            "instance": nn.Sequential(
                nn.Linear(instance_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            ),

            "coalition": nn.Sequential(
                nn.Linear(n_vehicles, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            ),

            "head": nn.Sequential(
                nn.Linear(256 + 256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        })

    def forward(self,
                instance: torch.Tensor,
                coalition: torch.Tensor):
        """"""
        instance = self.extractors["instance"](instance)
        coalition = self.extractors["coalition"](coalition)
        features = torch.cat(tensors = [instance, coalition], dim = 1)

        return self.extractors["head"](features)