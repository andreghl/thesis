from bargain.tests import test_response

import matplotlib.pyplot as plt
import numpy as np

shares = probs = np.zeros((3, 10000))
fig, axes = plt.subplots(nrows = 1, ncols = 2,
                         figsize = (12, 5))

for i in range(3):

    ax = axes[min(i, len(axes) - 1)]

    if i == 1:
        continue


    shares, probs = test_response(agent_path = "data/agents/1/bargain_5544",
                  proposer = 1,
                  agent = i,
                  time = 1,
                  fig = fig,
                  ax = ax)

plt.show()

agent = 0
plt.scatter(shares[:, agent], probs[agent, :])
plt.xlabel(f"Share to Agent {agent}")
plt.ylabel("Probability of Acceptance")
plt.show()