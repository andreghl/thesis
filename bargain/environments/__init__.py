from gymnasium.envs.registration import register, registry
from .utils import *


if "Selection_Discrete-v0" not in registry:
    from .selection import Selection
    register(
        id = "Selection_Discrete-v0",
        entry_point = Selection,
        max_episode_steps = 10)

if "Selection_Continuous-v0" not in registry:
    from .selection import SelectionDummy
    register(
        id = "Selection_Continuous-v0",
        entry_point = SelectionDummy,
        max_episode_steps = 10)

if "Selection_Continuous-v1" not in registry:
    from .selection import _Selection
    register(
        id = "Selection_Continuous-v1",
        entry_point = _Selection,
        max_episode_steps = 10)

if "Proposal-v0" not in registry:
    from .proposal import Proposal
    register(
        id = "Proposal-v0",
        entry_point = Proposal,
        max_episode_steps = 10
    )

if "Response_Discrete-v0" not in registry:
    from .response import Response
    register(
        id = "Response_Discrete-v0",
        entry_point = Response,
        max_episode_steps = 10)

if "Response_Continuous-v0" not in registry:
    from .response import _Response
    register(
        id = "Response_Continuous-v0",
        entry_point = _Response,
        max_episode_steps = 10)