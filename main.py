from bargain import train


seeds = [0]
timesteps: int = 1000000

for seed in seeds:

      ext: bool = True
      path = f'data/agents/bargain_{seed}'

      train(seed = seed,
            total_timesteps = timesteps,
            load_extractor = ext,
            agent_path = path)