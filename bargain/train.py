from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from bargain.agents import BargainPolicy
from stable_baselines3 import PPO
import gymnasium as gym
import bargain.env
import torch
import os

N_ENVS: int = 1
N_VEHICLES: int = 3
N_CUSTOMERS: int = 9
LOAD_EXTRACTOR: bool = True
NEW_AGENT: bool = True
LEARNING_RATE: float = 3e-5
GAMMA: float = 0.99
BATCH_SIZE: int = 256
N_EPOCHS: int = 3
FEATURES_DIM: int = 320
INSTANCE_SIZE: int = 40
ENT_COEF: float = 0.3
VF_COEF: float = 0.5
CLIP_RANGE: float = 0.2
TOTAL_TIMESTEPS: int = 500000
VERBOSE: int = 1
EXTRACTOR_PATH: str = "data/models/GainNN.pth"
AGENT_PATH: str = "data/agents/bargain"
LOG_PATH: str = "data/logs/agents/"


checkpoint = CheckpointCallback(save_freq = 1000,
                                save_path = "data/agents/checkpoints",
                                verbose = VERBOSE)
evaluation = EvalCallback(eval_env = gym.make(id = "Bargain-v0",
                                              n_vehicles = 3,
                                              n_customers = 9),
                          best_model_save_path = "data/agents/evaluation",
                          log_path = "data/logs/agents/eval",
                          eval_freq = 10000,
                          verbose = VERBOSE)

progress = ProgressBarCallback()
callback = CallbackList(
    [checkpoint, evaluation, progress]
)


def pretrain(seed: int,
             n_envs: int = N_ENVS,
             total_timesteps: int = TOTAL_TIMESTEPS,
             load_extractor: bool = LOAD_EXTRACTOR,
             create_new_agent: bool = NEW_AGENT,
             agent_path: str = AGENT_PATH,
             log_path: str = LOG_PATH,
             verbose: int = VERBOSE,
             **kwargs):
    """"""

    names = ["Responder-v0", "Proposer-v0"]


    for i, name in enumerate(names):

        _log_path = log_path + name
        env = make_vec_env(env_id = name,
                           n_envs = n_envs,
                           seed = seed,
                           env_kwargs = {
                               "n_vehicles": kwargs.pop("n_vehicles", N_VEHICLES),
                               "n_customers": kwargs.pop("n_customers", N_CUSTOMERS)
                           })

        if not os.path.exists(agent_path + ".zip") or create_new_agent:

            model = PPO(policy = BargainPolicy,
                        policy_kwargs = {
                            "features_dim": kwargs.pop("features_dim", FEATURES_DIM),
                            "instance_size": kwargs.pop("instance_size", INSTANCE_SIZE),
                            "n_vehicles": kwargs.pop("n_vehicles", N_VEHICLES),
                            "path": kwargs.pop("extractor_path", EXTRACTOR_PATH),
                            "load": load_extractor},
                        env = env,
                        learning_rate = kwargs.pop("learning_rate", LEARNING_RATE),
                        gamma = kwargs.pop("gamma", GAMMA),
                        clip_range = kwargs.pop("clip_range", CLIP_RANGE),
                        batch_size = kwargs.pop("batch_size", BATCH_SIZE),
                        n_epochs = kwargs.pop("n_epochs", N_EPOCHS),
                        ent_coef = kwargs.pop("ent_coef", ENT_COEF),
                        vf_coef = kwargs.pop("vf_coef", VF_COEF),
                        tensorboard_log = _log_path,
                        verbose = verbose,
                        seed = seed)

        else:

            model = PPO.load(path = agent_path,
                             policy_kwargs = {
                                 "features_dim": kwargs.pop("features_dim", FEATURES_DIM),
                                 "instance_size": kwargs.pop("instance_size", INSTANCE_SIZE),
                                 "n_vehicles": kwargs.pop("n_vehicles", N_VEHICLES),
                                 "path": kwargs.pop("extractor_path", EXTRACTOR_PATH),
                                 "load": load_extractor},
                             env = env,
                             n_epochs = kwargs.pop("n_epochs", N_EPOCHS),
                             ent_coef = kwargs.pop("ent_coef", ENT_COEF),
                             vf_coef = kwargs.pop("vf_coef", VF_COEF),
                             verbose = verbose,
                             seed = seed); print(f"> PPO agent loaded from {agent_path}")

        model.learn(total_timesteps = total_timesteps,
                    reset_num_timesteps = create_new_agent,
                    callback = callback,
                    progress_bar = True)
        model.save(agent_path)

    return None

def train(seed: int,
          env: str = 'Bargain-v0',
          n_envs: int = N_ENVS,
          total_timesteps: int = TOTAL_TIMESTEPS,
          load_extractor: bool = LOAD_EXTRACTOR,
          create_new_agent: bool = NEW_AGENT,
          agent_path: str = AGENT_PATH,
          log_path: str | None = LOG_PATH,
          verbose: int = VERBOSE,
          **kwargs):

    if os.path.exists(agent_path + ".zip"):
        print(f"File {agent_path} already exists...")
        return None

    message = f"""
> Running PPO:
    Seed: {seed}
    Extractor: {load_extractor}
    Timesteps: {total_timesteps}
    Environment: {env}
    Path: {agent_path}
    Logs: {log_path}
==================================
    """; print(message)

    env = make_vec_env(env_id = env,
                       n_envs = n_envs,
                       seed = seed,
                       env_kwargs = {
                           "n_vehicles": kwargs.pop("n_vehicles", N_VEHICLES),
                           "n_customers": kwargs.pop("n_customers", N_CUSTOMERS)
                       })

    if not os.path.exists(agent_path + ".zip") or create_new_agent:

        model = PPO(policy = BargainPolicy,
                    policy_kwargs = {
                        "features_dim": kwargs.pop("features_dim", FEATURES_DIM),
                        "instance_size": kwargs.pop("instance_size", INSTANCE_SIZE),
                        "n_vehicles": kwargs.pop("n_vehicles", N_VEHICLES),
                        "path": kwargs.pop("extractor_path", EXTRACTOR_PATH),
                        "load": load_extractor},
                    env = env,
                    learning_rate = kwargs.pop("learning_rate", LEARNING_RATE),
                    gamma = kwargs.pop("gamma", GAMMA),
                    clip_range = kwargs.pop("clip_range", CLIP_RANGE),
                    batch_size = kwargs.pop("batch_size", BATCH_SIZE),
                    n_epochs = kwargs.pop("n_epochs", N_EPOCHS),
                    ent_coef = kwargs.pop("ent_coef", ENT_COEF),
                    vf_coef = kwargs.pop("vf_coef", VF_COEF),
                    tensorboard_log = log_path,
                    verbose = verbose,
                    seed = seed)

    else:

        model = PPO.load(path = agent_path,
                         policy_kwargs = {
                             "features_dim": kwargs.pop("features_dim", FEATURES_DIM),
                             "instance_size": kwargs.pop("instance_size", INSTANCE_SIZE),
                             "n_vehicles": kwargs.pop("n_vehicles", N_VEHICLES),
                             "path": kwargs.pop("extractor_path", EXTRACTOR_PATH),
                             "load": load_extractor},
                         env = env,
                         n_epochs = kwargs.pop("n_epochs", N_EPOCHS),
                         ent_coef = kwargs.pop("ent_coef", ENT_COEF),
                         vf_coef = kwargs.pop("vf_coef", VF_COEF),
                         verbose = verbose,
                         seed = seed); print(f"> PPO agent loaded from {agent_path}")

    model.learn(total_timesteps = total_timesteps,
                reset_num_timesteps = create_new_agent,
                callback = callback,
                tb_log_name = f"PPO_{seed}")
    model.save(agent_path)

    return None