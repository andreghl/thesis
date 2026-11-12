from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from typing import Any
import pandas as pd
import numpy as np
import os

def plot(data: list,
         title: str,
         label: str,
         steps: list | None = None) -> None:
    """"""
    steps = range(len(data)) if steps is None else steps
    plt.plot(steps, data)
    plt.title(title)
    plt.xlabel("steps")
    plt.ylabel(label)
    plt.grid(True)
    plt.show()

    return None

def pad(data: list,
        n: int,
        value: Any = -1) -> list:
    """"""
    data +=  [value] * (n - len(data))
    return data

def unpad(data: list,
          value: Any = -1) -> list:
    """"""
    data = np.asarray(data)
    condition = data != value
    data = data[condition]

    return data.astype(float).tolist()

def ma(x, size = 1):
    w = np.ones(size) / size
    return np.convolve(x, w, mode = 'same')


categories = range(1, 13)

max_len = 196
n_steps = 400000
log_dir = "logs/agents/Bargain-v0"
names = ["eval/mean_reward",
         "eval/mean_ep_length",
         "rollout/ep_rew_mean",
         "rollout/ep_len_mean"]
file_path = "data/logs/agents/bargain.csv"

for _id in categories:

    path = f"{log_dir}/PPO_{_id}"
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    for name in names:

        data = [e.value for e in ea.Scalars(name)]
        col_name = f"{name}_{_id}"
        steps = [e.step for e in ea.Scalars(name)]
        col_steps_name = f"{name}_{_id}_steps"

        if not os.path.exists(file_path):
            df = pd.DataFrame()
        else:
            df = pd.read_csv(file_path)

        df[col_name] = pad(data, n = max_len)
        df[col_steps_name] = pad(steps, n = max_len)
        df.to_csv(file_path, index = False)

df = pd.read_csv(file_path)
choice = 3
variable = names[choice]
plot_names = [
    "Evaluation - Mean Reward",
    "Evaluation - Episode Length",
    "Training - Mean Reward",
    "Training - Episode Length"
]

for _n, (i, j, k) in enumerate(zip(range(1, 13, 3), range(2, 13, 3), range(3, 13, 3))):

    cols = [f'{variable}_{i}', f'{variable}_{j}', f'{variable}_{k}']
    cols_step = [f"{col}_steps" for col in cols]
    series = df[cols]
    mask = (series != -1).all(axis = 1)
    series = series[mask]
    steps = df[cols_step][mask].mean(axis = 1)
    series_m = series.mean(axis = 1)
    series_sd = series.std(axis = 1)


    plt.plot(steps, ma(series_m, 5), label = f"Type {_n}")
    plt.title(plot_names[choice])
    plt.legend()

plt.show()


"""
plot(data = unpad(df["eval/mean_reward_12"]),
     title = 'data',
     label = 'data',
     steps = unpad(df["eval/mean_reward_12_steps"]))
"""