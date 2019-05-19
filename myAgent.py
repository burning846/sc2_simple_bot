from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

class MoveToBeacon(base_agent.BaseAgent):

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        print(type(obs.observation))
        print(FUNCTIONS)

        time.sleep(0.5)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            return FUNCTIONS.no_op()
        else:
            return FUNCTIONS.select_army("select")