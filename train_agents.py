from bargain import train


seeds = [2910]
steps = 1000000

for seed in seeds:

    ext = True
    path = f"data/agents/{int(ext)}/bargain_{seed}"
    train(seed = seed,
          total_timesteps = steps,
          load_extractor = ext,
          agent_path = path)

    ext = False
    path = f"data/agents/{int(ext)}/bargain_{seed}"
    train(seed = seed,
          total_timesteps = steps,
          load_extractor = ext,
          agent_path = path)