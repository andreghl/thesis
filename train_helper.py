from bargain.networks import GainNN
import bargain.networks as net


params = {"seed": 0,
          "learning_rate": 1e-4,
          "weight_decay": 5e-5,
          "batch_size": 128}

print(f"Training Gain network with seed {params['seed']}...")
net.train(**params,
          model = GainNN(),
          n_epochs = 20,
          data_path = "data/instances.h5",
          features = ["instance", "coalitions"],
          target = "gain",
          label = "GainNN",
          verbose = 1)