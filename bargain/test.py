from bargain.tests import test_agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import bargain.env
import json

n_episodes = 1000
max_time = 9

"""
test_agent(test_env = "Bargain-v0",
           agent_path = "data/agents/1/bargain_3241",
           data_path = "data/tests/",
           n_episodes = n_episodes)
"""

with open("data/tests/test_agent.json") as f:
    data = json.load(f)

data = pd.DataFrame.from_dict(data)

time = np.zeros(n_episodes)
coalition = np.zeros(shape = (3, n_episodes * 10))
proposal = np.zeros(shape = (3, n_episodes * 10))
response = np.zeros(shape = (3, n_episodes * 10))

agreement = np.zeros(shape = (3, n_episodes))
truncated = np.zeros(n_episodes)

shapley = np.zeros(shape = (3, n_episodes))
nucleolus = np.zeros(shape = (3, n_episodes))

for e in range(n_episodes):

    times = data[str(e)].loc['time']
    time[e] = len(times)
    truncated[e] = data[str(e)].loc['truncated']

    if not truncated[e]:
        agreement[:, e] = np.asarray(data[str(e)].loc['proposal'][-1])
    else:
        agreement[:, e] = np.zeros(3)

    for k, t in enumerate(times):
        x = e + (max_time - t)
        coalition[:, x] = np.asarray(data[str(e)].loc['coalition'][k])
        proposal[:, x] = np.asarray(data[str(e)].loc['proposal'][k])
        response[:, x] = np.asarray(data[str(e)].loc['response'][k])
        shapley[:, e] = np.asarray(data[str(e)].loc['shapley'])
        nucleolus[:, e] = np.asarray(data[str(e)].loc['nucleolus'])

sns.histplot(time, bins = 6)
plt.show()

valid = np.logical_not(truncated)

print("Means")
print('coalitions', coalition.mean(axis = -1, where = proposal > 0))
print('responses', response.mean(axis = -1, where = proposal > 0))
print('proposals', proposal.mean(axis = -1, where = proposal > 0))
print('agreements', agreement.mean(axis = -1, where = valid))
print('shapley', shapley.mean(axis = -1))
print('nucleolus', nucleolus.mean(axis = -1))
print('truncated:', sum(truncated) / len(truncated))

agent = 1
plt.scatter(shapley[agent, valid], agreement[agent, valid])
plt.xlabel('Shapley value')
plt.ylabel('Proposed share')
plt.show()

plt.scatter(nucleolus[agent, valid], agreement[agent, valid])
plt.xlabel('nucleolus')
plt.ylabel('Proposed share')
plt.show()