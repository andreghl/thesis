from bargain import generate_observation
from stable_baselines3 import PPO

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import bargain.env
import torch



def test_response(agent_path: str,
                  proposer: int,
                  agent: int,
                  time: int,
                  n: int = 10000,
                  fig = None,
                  ax = None):
    """"""
    instance, *_ = generate_observation(n_vehicles = 3, n_customers = 9,radius = 1.0, seed = 2026)

    env = gym.make("Bargain-v0",
                   n_vehicles = 3,
                   n_customers = 9,
                   random_start = False); env.reset()


    model = PPO.load(agent_path)

    shares = np.random.dirichlet(alpha = [0.5, 0.5, 0.5], size = n)
    env.unwrapped.change_instance(instance, proposer = proposer)
    obs = env.unwrapped.obs
    obs = {k: np.repeat(np.asarray(v)[None, ...], n, axis = 0) for k, v in obs.items()}
    obs["proposal"] = shares.astype(np.float32)

    probs = np.zeros((3, n))

    with torch.no_grad():

        for i in range(3):

            obs["agent"] = np.ones(n) * i
            obs, _ = model.policy.obs_to_tensor(obs)

            dist = model.policy.get_distribution(obs)
            probs[i] = dist.dist_response.probs.squeeze(-1).cpu().numpy()

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    plot = ax.tripcolor(shares[:, proposer], shares[:, agent], probs[agent, :],
                        shading = "gouraud", cmap = "viridis",
                        vmin = probs.min(), vmax = probs.max())
    cbar = fig.colorbar(plot, ax = ax)
    cbar.set_label(f"P(Accept | p = {proposer}, a = {agent}, t = {time})")
    plt.xlabel(f"Agent {proposer}")
    plt.ylabel(f"Agent {agent}")


    return shares, probs