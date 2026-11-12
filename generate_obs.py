from bargain import generate_observations

seed: int = 0
print(f"Generating observations with seed {seed}...")
generate_observations(filename = "data/instances.h5",
                      obs = 1000000,
                      vehicles = 3,
                      customers = 9,
                      radius = 1.0,
                      seed = seed)