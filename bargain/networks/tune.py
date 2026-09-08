from .train import train
import torch.nn as nn
import numpy as np
import heapq
import copy



def tune(model: nn.Module,
         parameters: dict[str, tuple[float, float]],
         n_models: int,
         tune_epochs: int,
         data_path: str,
         features: list[str],
         target: str,
         seed: int = 0,
         label: str = "model",
         log_dir: str = "data/logs/tuning"):

    queue = []
    heapq.heapify(queue)
    MODEL = model

    for i in range(n_models):

        params = {'seed': seed}

        for key, (low, high) in parameters.items():
            if key == "batch_size":
                params[key] = 2 * int(np.random.uniform(int(low / 2), int(high / 2)))
                continue

            params[key] = np.random.uniform(low, high)

        _, score = train(**params,
                         model = copy.deepcopy(MODEL),
                         n_epochs = tune_epochs,
                         data_path = data_path,
                         features = features,
                         target = target,
                         label = label,
                         log_dir = log_dir,
                         verbose = 0,
                         log_every = None)

        heapq.heappush(queue, (score, params))

    return heapq.heappop(queue)