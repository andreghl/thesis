from gymnasium.envs.registration import register, registry

max_env_steps = 9
respond_env_steps = 4
agent_steps = 3
reward_steps = 3
"""
The env must loop through all agents to get their respective action
=> Approximate MARL with single RL env.
and the env must then loop through all the agents to assign the rewards.
Default n_agents = 3
"""
max_episode_steps = max_env_steps * (agent_steps + reward_steps)

if "Proposer-v0" not in registry:
    from .dummies.proposer import Proposer
    register(id = "Proposer-v0",
             entry_point = Proposer,
             max_episode_steps = max_env_steps)

if "Responder-v0" not in registry:
    from .dummies.responder import Responder
    register(id = "Responder-v0",
             entry_point = Responder,
             max_episode_steps = max_env_steps * respond_env_steps)

if "Bargain-v0" not in registry:
    from .main import Bargain
    register(id = "Bargain-v0",
             entry_point = Bargain,
             max_episode_steps = max_episode_steps)
