from stable_baselines3.common.env_util import make_vec_env
from bargain.env.dummies import mask
from stable_baselines3 import PPO
from tqdm import tqdm
import numpy as np
import bargain.env
import json
import os


def test_agent(test_env: str,
               agent_path: str,
               data_path: str,
               n_episodes: int,
               seed: int = 2026,
               n_vehicles: int = 3,
               n_customers: int = 9):

    n_envs = 1
    env = make_vec_env(env_id = test_env,
                       n_envs = n_envs,
                       seed = seed,
                       env_kwargs = {
                           'n_vehicles': n_vehicles,
                           'n_customers': n_customers,
                           'random_start': False
                       })

    model = PPO.load(agent_path)
    dataset = {episode: {} for episode in range(n_episodes)}

    for episode in tqdm(range(n_episodes)):

        state = env.reset()
        done = np.full(shape = n_envs,
                        fill_value = False)

        data = {
            'time': [],
            'proposer': [],
            'agent': [],
            'proposal': [],
            'coalition': [],
            'response': []
        }

        act = ['coalition', 'proposal', 'response']
        info = {}

        while not np.any(done):

            action, _ = model.predict(state)
            next_state, reward, done, info = env.step(action)

            state['action']  = action[:, mask(*action, *state["role"], n_vehicles)]

            if state['role'] == 0:
                for key in set(data.keys()) - set(act):
                    data[key].append(state[key]
                                          .astype(int)
                                          .tolist()[0])

                for key in act:
                    if key != 'response':
                        data[key].append(*state[key]
                                              .astype(float)
                                              .tolist())

            if state['role'] == 1 and next_state['role'] == 2:
                data['response'].append(*state['response']
                                      .astype(float)
                                      .tolist())


            state = next_state.copy()

        shapley = env.envs[0].env.unwrapped.shapley
        if np.sum(shapley) > 0 and np.all(shapley >= 0):
            shapley = shapley / np.sum(shapley)
        else:
            shapley = np.zeros(n_vehicles)

        nucleolus = env.envs[0].env.unwrapped.nucleolus
        if np.sum(nucleolus) > 0 and np.all(nucleolus >= 0):
            nucleolus = nucleolus / np.sum(nucleolus)
        else:
            nucleolus = np.zeros(n_vehicles)

        for i in info:
            info = i

        data['shapley'] = shapley.astype(float).tolist()
        data['nucleolus'] = nucleolus.astype(float).tolist()
        data['truncated'] = info.pop('TimeLimit.truncated', False)
        # data['_shapley'] = env.envs[0].env.unwrapped.shapley.astype(float).tolist(),
        # data['_nucleolus'] = env.envs[0].env.unwrapped.nucleolus.astype(float).tolist()
        dataset[episode] = data

        gains = env.envs[0].env.unwrapped.gains
        data['gains'] = gains.astype(float).tolist()

    os.makedirs(data_path, exist_ok = True)
    with open(data_path + "test_agent.json", "w", encoding = "utf-8") as f:
        json.dump(dataset, f, indent = 2, ensure_ascii = False)

    return None