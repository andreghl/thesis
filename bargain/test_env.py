import gymnasium as gym
import bargain.env

max_rounds = 10
env_steps = 6
max_steps = env_steps * max_rounds - 1

env = gym.make(id = "Bargain-v0",
                max_episode_steps = max_steps,
                n_vehicles = 3,
                n_customers = 9); env.reset()

done = False
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print({k: v for k, v in state.items() if k not in ['instance']})
    print(reward, terminated, truncated, info)
    print("\n")

