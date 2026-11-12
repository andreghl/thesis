from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib



def ma(x, size = 1):
    w = np.ones(size) / size
    return np.convolve(x, w, mode = 'same')


def plot_tensorboard(path: str,
                     variables: list[str],
                     xlabel: str | None = None,
                     ylabel: str | None = None,
                     label: str | None = None,
                     ma_size: int = 1,
                     data: pd.DataFrame | None = None,
                     ax : matplotlib.axes.Axes | None = None):

    data = pd.DataFrame() if data is None else data
    xlabel = xlabel or ""
    ylabel = ylabel or ""
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    ax = ax or plt.gca()

    for variable in variables:

        name = label or variable
        data['steps'] = [e.step for e in ea.Scalars(variable)]
        data[f'{name}'] = [e.value for e in ea.Scalars(variable)]

        ax.plot(data['steps'],
                 ma(data[f'{name}'], size = max(ma_size, 1)),
                 label = f"{name}")

        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return data

fig, ((one, two), (three, four)) = plt.subplots(nrows = 2,
                                            ncols = 2,
                                            figsize = (12, 10))

data = plot_tensorboard(path = "logs/agents/PPO_3241_1",
                        variables = ["eval/mean_reward"],
                        label = '3241',
                        ax = one)
data = plot_tensorboard(path = "logs/agents/PPO_5544_1",
                        variables = ["eval/mean_reward"],
                        label = '5544',
                        data = data,
                        ax = one)
data = plot_tensorboard(path = "logs/agents/PPO_7891_1",
                        variables = ["eval/mean_reward"],
                        label = '7891',
                        xlabel = 'Environment Step (Helper)',
                        ylabel = 'Mean Episode Reward',
                        data = data,
                        ax = one)

data = plot_tensorboard(path = "logs/agents/PPO_3241_1",
                        variables = ["eval/mean_ep_length"],
                        label = '3241',
                        data = data,
                        ax = two)
data = plot_tensorboard(path = "logs/agents/PPO_5544_1",
                        variables = ["eval/mean_ep_length"],
                        label = '5544',
                        data = data,
                        ax = two)
data = plot_tensorboard(path = "logs/agents/PPO_7891_1",
                        variables = ["eval/mean_ep_length"],
                        label = '7891',
                        xlabel = 'Environment Step (Helper)',
                        ylabel = 'Mean Episode Length',
                        data = data,
                        ax = two)

data = plot_tensorboard(path = "logs/agents/PPO_3241_2",
                        variables = ["eval/mean_reward"],
                        label = '3241',
                        ax = three)
data = plot_tensorboard(path = "logs/agents/PPO_5544_2",
                        variables = ["eval/mean_reward"],
                        label = '5544',
                        data = data,
                        ax = three)
data = plot_tensorboard(path = "logs/agents/PPO_7891_2",
                        variables = ["eval/mean_reward"],
                        label = '7891',
                        xlabel = 'Environment Step (no Helper)',
                        ylabel = 'Mean Episode Reward',
                        data = data,
                        ax = three)

data = plot_tensorboard(path = "logs/agents/PPO_3241_2",
                        variables = ["eval/mean_ep_length"],
                        label = '3241',
                        data = data,
                        ax = four)
data = plot_tensorboard(path = "logs/agents/PPO_5544_2",
                        variables = ["eval/mean_ep_length"],
                        label = '5544',
                        data = data,
                        ax = four)
data = plot_tensorboard(path = "logs/agents/PPO_7891_2",
                        variables = ["eval/mean_ep_length"],
                        label = '7891',
                        xlabel = 'Environment Step (no Helper)',
                        ylabel = 'Mean Episode Length',
                        data = data,
                        ax = four)

print(data)
# plt.fill_between(t, mean - std, mean + std, alpha=0.3, label="±1 std")
plt.legend()
plt.show()