from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from pysc2.lib import units

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_COMMAND_CENTER = units.Terran.CommandCenter
_VESPENE_GEYSER = units.Neutral.VespeneGeyser
FUNCTIONS = actions.FUNCTIONS

# Parameters
_NOT_QUEUED = [0]
_QUEUED = [1]

_MINERAL_COLLECTED = 1
_FOOD_USED = 3
_FOOD_CAP = 4

_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class FindAndDefeatZerglings(base_agent.BaseAgent):
    """An agent specifically for solving the DefeatRoaches map."""

    def reset(self):
        super(FindAndDefeatZerglings, self).reset()
        self.loc = [[14,25],[14,34],[14,45],[26,45],[38,45],[50,45],[50,34],[50,25],[38,25],[38,34],[26,34],[26,25]]
        self.index = len(self.loc) - 3

    def step(self, obs):

        if FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
            self.index = (self.index + 1) % len(self.loc)
            return FUNCTIONS.Attack_minimap("queued", self.loc[self.index])
        else:
            return FUNCTIONS.select_army("select")
