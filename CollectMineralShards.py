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


def get_random_position(obs, typename):
    for unit in obs.observation.feature_units:
        if unit.unit_type == typename and int(unit.build_progress) == 100:
            return tuple((unit.x, unit.y))

    return None


def get_idle_position(obs, typename):
    for unit in obs.observation.feature_units:
        if unit.unit_type == typename and int(unit.order_length) == 0 and int(unit.build_progress) == 100:
            return tuple((unit.x, unit.y))
    return None


def get_num(obs, typename):
    cnt = 0
    for unit in obs.observation.feature_units:
        if unit.unit_type == typename:
            cnt += 1
    return cnt


def _get_minerals_positions(obs):
    res = []
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Neutral.MineralField:
            res.append(tuple((unit.x, unit.y)))
    return res


def check_selected(obs, pos):
    for units in obs.observation.feature_units:
        if units.is_selected != 0 and units.x == pos[0] and units.y == pos[1]:
            return True

    return False


def TSP(marine_xy, minerals, used, cnt):
    # print(cnt)
    if cnt < 1:
        return 0, 0
    total_cost = 9999999999
    idx = 0
    for i in range(len(minerals)):
        if used[i] == 1:
            continue
        distance = numpy.linalg.norm(
            numpy.array(minerals[i]) - numpy.array(marine_xy), axis=0)
        used[i] = 1
        cost, _ = TSP(marine_xy, minerals, used, cnt - 1)
        used[i] = 0
        if cost + distance < total_cost:
            total_cost = cost + distance
            idx = i

    return total_cost, idx



class CollectMineralShards(base_agent.BaseAgent):

    def setup(self, obs_spec, action_spec):
        super(CollectMineralShards, self).setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent requires the feature_units observation.")

    def reset(self):
        super(CollectMineralShards, self).reset()
        self._marine_selected = False
        self._previous_mineral_xy = [-1, -1]

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)
        marines = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_SELF]
        if not marines:
            return FUNCTIONS.no_op()
        marine_unit = next((m for m in marines
                            if m.is_selected == self._marine_selected), marines[0])
        marine_other = next((m for m in marines
                            if m.is_selected != self._marine_selected), marines[1])
        marine_xy = [marine_unit.x, marine_unit.y]

        if marine_unit.x < marine_other.x:
            left = 1
        else:
            left = 0

        if not marine_unit.is_selected:
            # Nothing selected or the wrong marine is selected.
            self._marine_selected = True
            return FUNCTIONS.select_point("select", marine_xy)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # Find and move to the nearest mineral.
            if left:
                minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                          if unit.alliance == _PLAYER_NEUTRAL and unit.x <= 42]
            else:
                minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                            if unit.alliance == _PLAYER_NEUTRAL and unit.x > 42]


            if minerals:
                # Find the best
                used = [0 for _ in minerals]
                # cost, idx = TSP(marine_xy, minerals, used, len(minerals))

                # Find the closest.
                distances = numpy.linalg.norm(
                    numpy.array(minerals) - numpy.array(marine_xy), axis=1)
                closest_mineral_xy = minerals[numpy.argmin(distances)]

                # Swap to the other marine.
                self._marine_selected = False
                return FUNCTIONS.Move_screen("now", closest_mineral_xy)
            else:
                self._marine_selected = False
                return FUNCTIONS.no_op()

        return FUNCTIONS.no_op()