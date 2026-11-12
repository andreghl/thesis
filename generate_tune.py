from bargain import generate_observations

# ensure that the seed here is different from the seed in 'generate_obs.py'
seed: int = 2026
print(f"Generating observations for hyperparameter tuning with seed {seed}...")
generate_observations(filename = "data/tune.h5",
                      obs = 100000,
                      vehicles = 3,
                      customers = 9,
                      radius = 1.0,
                      seed = seed)