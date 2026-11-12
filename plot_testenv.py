from bargain.tests import test_agent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

N: int = 10000


test_agent(test_env = "Bargain-v0",
           agent_path = "data/agents/1/bargain_2910",
           data_path = "data/tests/",
           n_episodes = N,
           seed = 2526)


with open("data/tests/test_agent.json") as f:
    data = json.load(f)
    data = pd.DataFrame.from_dict(data)

time = np.zeros(N)

coalition = np.zeros(shape = (3, N))
proposal = np.zeros(shape = (3, N))
response = np.zeros(shape = (3, N))
shapley = np.zeros(shape = (3, N))
nucleolus = np.zeros(shape = (3, N))
truncated = np.zeros(shape = N)
superadditive = np.ones(shape = N)

selected = np.zeros(shape = (N, 4))
selection = np.zeros(shape = N)
gains = np.zeros(shape = (N, 4))

coal_share = {(1, 1, 0): 0,
              (1, 0, 1): 0,
              (0, 1, 1): 0,
              (1, 1, 1): 0}

not_sadd = {(1, 1, 0): 0,
              (1, 0, 1): 0,
              (0, 1, 1): 0,
              (1, 1, 1): 0}

for e in range(N):

    _times = data[str(e)].loc['time']
    time[e] = len(_times)
    truncated[e] = data[str(e)].loc['truncated']

    shapley[:, e] = np.asarray(data[str(e)].loc['shapley'])
    nucleolus[:, e] = np.asarray(data[str(e)].loc['nucleolus'])

    if not truncated[e]:
        _coalition = data[str(e)].loc['coalition'][-1]
        _proposal = data[str(e)].loc['proposal'][-1]
        _response = data[str(e)].loc['response'][-1]

        _gains = data[str(e)].loc['gains']
        gains[e] = _gains
        k = np.argmax(_gains)

        for i, coal in enumerate(coal_share):

            if _gains[i] > _gains[-1]:
                superadditive[e] = 0

            if tuple(_coalition) == coal:
                coal_share[coal] += 1
                selected[e, i] = 1
                selection[e] = i

                if _gains[i] > _gains[-1]:
                    not_sadd[coal] += 1

        coalition[:, e] = np.asarray(_coalition)
        proposal[:, e] = np.asarray(_proposal)
        response[:, e] = np.asarray(_response)
    else:
        coalition[:, e] = np.zeros(3)
        proposal[:, e] = np.zeros(3)
        response[:, e] = np.zeros(3)

plt.hist(time, bins = np.arange(9), align = 'left')
plt.show()

fig, (left, right) = plt.subplots(1, 2,
                                  figsize = (12, 5))

agent = 1
valid = np.logical_not(truncated) & coalition[agent].astype(bool)
left.scatter(shapley[agent, valid], proposal[agent, valid])
left.set_xlabel('Shapley value')
left.set_ylabel('Proposed share')

right.scatter(nucleolus[agent, valid], proposal[agent, valid])
right.set_xlabel('nucleolus')
right.set_ylabel('Proposed share')
plt.show()


_, axes = plt.subplots(1, 3,
                                       figsize = (18, 5))
k = 0.8

labels = ['proposal', 'shapley', 'nucleolus']
bins = np.linspace(0, 1, 21)

for agent, ax in enumerate(axes):
    valid = np.logical_not(truncated) & (coalition.sum(axis = 0) == 3)
    dfs = [proposal[agent, valid], shapley[agent, valid], nucleolus[agent, valid]]

    for df, label in zip(dfs, labels):
        ax.hist(df,
                label = label,
                alpha = k,
                bins = bins,
                histtype = "step",
                linewidth = 2)
        ax.set_xlabel(f'Agent {agent}')
        ax.set_ylabel('Count')

plt.legend()
plt.show()


coal_share = {key: value / N for key, value in coal_share.items()}
print('coal_share', coal_share)
unique, counts = np.unique(time, return_counts = True)
print('unique', unique)
print('counts', counts)
print('superadditive', (N - superadditive.sum()) / N)
print(not_sadd)

test = selected * gains
z = 0
print("value of selected", test.sum(axis = z) / selected.sum(axis = z))

test = np.abs(selected - 1) * gains
print("value not selected", test.sum(axis = z) /  np.abs(selected - 1).sum(axis = z))


superadditive = gains - gains[:, -1:]
superadditive = np.any(superadditive > 0, axis = 1)
print('not sa', superadditive.sum() / len(gains))

colors = ["tab:red", "tab:orange", "tab:blue"]

fig, axes = plt.subplots(2, 2,
                                  figsize = (12, 10))

"""
agent = 1
for i, ax in enumerate(axes):

    left, right = ax
    test_agent(test_env = "Bargain-v0",
               agent_path = f"data/agents/{1 - i}/bargain_5544",
               data_path = "data/tests/",
               n_episodes = N,
               seed = 2526)

    valid = np.logical_not(truncated) & coalition[agent].astype(bool)
    conds = [valid, superadditive, np.logical_not(valid)]
    names = ['valid', 'not superadditive', 'invalid']
    for i, (label, cond) in enumerate(zip(names, conds)):


        left.scatter(shapley[agent, cond], proposal[agent, cond], c = colors[i])
        left.set_xlabel('Shapley value')
        left.set_ylabel('Proposed share')

        right.scatter(nucleolus[agent, cond], proposal[agent, cond], c = colors[i], label = label)
        right.set_xlabel('nucleolus')
        right.set_ylabel('Proposed share')
        plt.legend()

plt.show()
"""

valid = coalition.sum(axis = 0) == 3
print('proposal', proposal.mean(axis = 1))
print('proposal', proposal[:, valid].mean(axis = 1))
print('nucleolus', nucleolus[:, valid].mean(axis = 1))
print('shapley', shapley[:, valid].mean(axis = 1))