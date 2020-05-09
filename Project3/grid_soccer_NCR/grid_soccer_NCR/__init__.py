import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Grid-Soccer-NCR-v1',
    entry_point='grid_soccer_NCR.envs:grid_soccer_NCR',
#    timestep_limit=1000,
#    reward_threshold=100.0,
#    nondeterministic = True,
)
