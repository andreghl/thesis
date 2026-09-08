from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from bargain.networks import GainNN, get_device
import gymnasium as gym
import torch.nn as nn
import torch
import errno
import os


class GainExtractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_dim: int = 320,
                 instance_size: int = 40,
                 n_vehicles: int = 3,
                 path: str | None = None,
                 load: bool = True,
                 device: torch.device = get_device()):
        super().__init__(observation_space, features_dim)

        self.instance_size = instance_size
        self.n_vehicles = n_vehicles
        self.device = device

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

            "features": nn.Sequential(
                nn.Linear(25, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            ),

            "fusion": nn.Sequential(
                nn.Linear(512 + 64, 320),
                nn.ReLU(),
                nn.Linear(320, features_dim),
                nn.ReLU()
            )
        })

        if path is not None and load:
                self.load(path)

    def forward(self, obs: gym.spaces.Dict):

        def ensure_dim(x: torch.Tensor) -> torch.Tensor:
            """Ensure that the tensor has two dimensions."""
            if x.dim() == 3 and x.size(1) == 1:
                return x.squeeze(1)
            return x

        instance = self.extractors["instance"](ensure_dim(obs["instance"]))
        coalition = self.extractors["coalition"](ensure_dim(obs["coalition"]))

        features = torch.cat([
            ensure_dim(obs["proposal"]),
            ensure_dim(obs["response"]),
            ensure_dim(obs["time"]),
            ensure_dim(obs["proposer"]),
            ensure_dim(obs["agent"]),
            ensure_dim(obs["role"])],
            dim = 1)

        features = self.extractors["features"](features)
        fusion = torch.cat([
            ensure_dim(instance),
            ensure_dim(coalition),
            features
        ], dim = 1)

        return self.extractors["fusion"](fusion)

    def load(self, path: str):

        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

        model = GainNN(self.instance_size, self.n_vehicles)
        model.load_state_dict(
            torch.load(f = path,
                       map_location = self.device,
                       weights_only = True)
        )

        common = set(self.extractors.keys()) & set(model.extractors.keys())
        for key in common:
            self.extractors[key].load_state_dict(
                model.extractors[key].state_dict()
            )

        print(f"> GainNN loaded from {path}")