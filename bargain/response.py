from stable_baselines3.common.env_util import make_vec_env
from bargain.agents import BargainPolicy
from stable_baselines3 import PPO
import envs

name = "Response-v0"
log_path = f"data/logs/agents/{name}"
agent_path = f"data/agents/{name}"

env = make_vec_env(env_id = name,
                   n_envs = 1,
                   env_kwargs = {
                       "n_vehicles": 3,
                       "n_customers": 9
                   })

model = PPO(policy = BargainPolicy,
            env = env,
            learning_rate = 1e-4,
            gamma = 0.99,
            batch_size = 256,
            n_epochs = 10,
            verbose = 1,
            tensorboard_log = log_path,
            policy_kwargs = {
                "features_dim": 320,
                "instance_size": 40,
                "n_vehicles": 3,
                "path": 'data/models/GainNN.pth',
                "load": False},
            ent_coef = 0.05)

model.learn(total_timesteps = 50000)
model.save(agent_path)

"""
PPO_1 = {lr: 2e-4, load = True}
PPO_2 = {lr: 2e-4, load = False}
PPO_3 = {lr: 3e-4, load = True}
PPO_4 = {lr: 3e-4, load = False}
PPO_5 = {lr: 1e-4, load = True}
PPO_6 = {lr: 1e-4, load = False}
"""