from bargain import generate_observations, train, pretrain
from bargain.networks import GainNN
from bargain.utils import save_params
import bargain.networks as net
import random

seed: int = 0
a, b = 0, 2026
random.seed(seed)
n_runs: int = 3
load_ext: bool = True
total_timesteps: int = 20000


print("Generating observations...")
generate_observations(filename = "data/instances.h5",
                      obs = 10000,
                      vehicles = 3,
                      customers = 9,
                      radius = 1.0,
                      seed = random.randint(a, b))

print("Generating observations for hyperparameter tuning...")
generate_observations(filename = "data/tune.h5",
                      obs = 1000,
                      vehicles = 3,
                      customers = 9,
                      radius = 1.0,
                      seed = random.randint(a, b))

print("Parameters to tune for the neural network: ")
parameters = {
    "learning_rate": (1e-5, 1e-1),
    "weight_decay": (0.0, 1e-3),
    "batch_size": (32, 180)}

for key, value in parameters.items():
    print(f"> {key}: {value}")

print("Tuning Gain network...")
score, params = net.tune(model = GainNN(),
                     parameters = parameters,
                     n_models = 15,
                     data_path = "data/tune.h5",
                     features = ["instance", "coalitions"],
                     target = "gain",
                     label = "GainNN",
                     tune_epochs = 3,
                     seed = random.randint(a, b))

print(f"Selected parameters: {params} with score {score}")
save_params(params = params,
            score = score,
            model_name = "GainNN")

print("Training Gain network...")
net.train(**params,
      model = GainNN(),
      n_epochs = 30,
      data_path = "data/instances.h5",
      features = ["instance", "coalitions"],
      target = "gain",
      label = "GainNN",
      verbose = 1,
      seed = random.randint(a, b))


print(f"Running PPO agent for {n_runs} with load_extractor: {load_ext}...")
for run in range(n_runs):

    pretrain(seed = random.randint(a, b),
             total_timesteps = total_timesteps,
             load_extractor = load_ext)
    train(seed = random.randint(a, b),
          total_timesteps = total_timesteps,
          load_extractor = load_ext)


load_ext = False
print(f"Running PPO agent for {n_runs} with load_extractor: {load_ext}...")
for run in range(n_runs):

    pretrain(seed = random.randint(a, b),
             total_timesteps = total_timesteps,
             load_extractor = load_ext)
    train(seed = random.randint(a, b),
          total_timesteps = total_timesteps,
          load_extractor = load_ext)