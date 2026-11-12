from bargain.networks import GainNN
from bargain.utils import save_params
import bargain.networks as net


seed: int = 0
print("Parameters to tune for the neural network: ")
parameters = {
    "learning_rate": (1e-5, 1e-1),
    "weight_decay": (0.0, 1e-3),
    "batch_size": (64, 256)}

for key, value in parameters.items():
    print(f"> {key}: {value}")

print(f"Tuning Gain network with seed {seed}...")
score, params = net.tune(model = GainNN(),
                         parameters = parameters,
                         n_models = 20,
                         data_path = "data/tune.h5",
                         features = ["instance", "coalitions"],
                         target = "gain",
                         label = "GainNN",
                         tune_epochs = 10,
                         seed = seed)

print(f"Selected parameters: {params} with score {score}")
save_params(params = params,
            score = score,
            model_name = "GainNN")