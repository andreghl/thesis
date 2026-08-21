from gymnasium.envs.registration import register, registry

if "Selection-v0" not in registry:
    from .dummies.selection import Selection
    register(id = "Selection-v0",
             entry_point = Selection,
             max_episode_steps = 10)

if "Proposal-v0" not in registry:
    from .dummies.proposal import Proposal
    register(id = "Proposal-v0",
             entry_point = Proposal,
             max_episode_steps = 10)

if "Response-v0" not in registry:
    from .dummies.response import Response
    register(id = "Response-v0",
             entry_point = Response,
             max_episode_steps = 10)