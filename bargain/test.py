from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from env.dummies import mask
import numpy as np
import bargain.env


name = "Bargain-v0"
agent_path = f"data/agents/bargain"
n_envs = 1
n_vehicles = 3

env = make_vec_env(env_id = name,
                   n_envs = n_envs,
                   env_kwargs = {
                       "n_vehicles": n_vehicles,
                       "n_customers": 9
                   })

test = PPO.load(agent_path)
state = env.reset()

dones = np.full(shape = n_envs, fill_value = False)

while not np.any(dones):

    agents = state["agent"]
    times = state["time"]
    roles = state["role"]
    _obs = state.copy()
    print({k: v for k, v in state.items() if k not in ["instance", "agent", "role", "time"]})
    actions, next_state = test.predict(state)
    state, rewards, dones, info = env.step(actions)

    actions = actions[:, mask(*actions, *roles, n_vehicles)]
    if len(actions) == 0:
        actions = [-1]

    for i in info:
        if "episode" in i.keys():
            print(i["episode"])


    print(f"{'Time':<5} {'Agent':<5} {'Role':<5} {'Action':<40} {'Reward':<5} {'Done':<5}")
    print("-" * 75)

    for time, agent, role, action, reward, done in list(zip(times, agents, roles, actions, rewards, dones)):
        print(f"{time:<5} {agent:<5} {role:<5} {str(action):<40} {reward:<5} {done:<5}")

    print("\n")

for i in info:
    info = i

print("\n")
print(f"Terminal observation: (Truncated: {info['TimeLimit.truncated']})")
print("=" * 70)
print({k: v for k, v in info["terminal_observation"].items() if k not in ["instance", "time", "agent", "role"]})
print(info["episode"])

shapley = env.envs[0].env.unwrapped.shapley
print("shapley", shapley / np.sum(shapley))
nucleolus = env.envs[0].env.unwrapped.nucleolus
print("nucleolus", nucleolus / np.sum(nucleolus))