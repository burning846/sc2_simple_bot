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


def _xy_command_center_point(obs):
    # commend_center_points = _xy_locs(obs.observation.feature_screen.unit_type == _COMMAND_CENTER)
    # commend_center_point = numpy.mean(commend_center_points, axis=0).round()
    # return commend_center_point
    for unit in obs.observation.feature_units:
        # print(int(unit.order_length) == 0)
        if unit.unit_type == units.Terran.CommandCenter:
            return tuple((unit.x, unit.y))

def _xyr_command_center_point(obs):
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.CommandCenter:
            return tuple((unit.x, unit.y, unit.radius))


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



def _get_potential_supply_positions(obs):
    res = []
    x, y, r = _xyr_command_center_point(obs)
    x = x + r - 26
    res.extend([(x, y + 25), (x, y - 25)])
    x = x + 9
    res.extend([(x, y - 16), (x, y + 16), (x, y + 25), (x, y - 25)])
    x = x + 8
    res.extend([(x, y - 16), (x, y + 16), (x, y + 25), (x, y - 25)])
    x = x + 9
    res.extend([(x, y - 16), (x, y + 16), (x, y + 25), (x, y - 25)])
    x = x + 8
    res.extend([(x, y - 16), (x, y + 16), (x, y - 7), (x, y + 7), (x, y + 25), (x, y - 25)])
    return res


def _get_potential_barrack_positions(obs):
    res = []
    x, y, r = _xyr_command_center_point(obs)
    x = x + r + 20
    res.extend([(x, y - 20), (x, y - 6), (x, y + 8), (x, y + 22)])
    x = x + 13
    res.extend([(x, y - 20), (x, y - 6), (x, y + 8)])
    return res


def _get_minerals_positions(obs):
    res = []
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Neutral.MineralField:
            res.append(tuple((unit.x, unit.y)))
    return res


def cal_food(obs):
    food = 0
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.CommandCenter:
            food += 15
        elif unit.unit_type == units.Terran.SupplyDepot:
            food += 8
    return food


def check_selected(obs, pos):
    for units in obs.observation.feature_units:
        if units.is_selected != 0 and units.x == pos[0] and units.y == pos[1]:
            return True

    return False


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
        marine_xy = [marine_unit.x, marine_unit.y]

        if not marine_unit.is_selected:
            # Nothing selected or the wrong marine is selected.
            self._marine_selected = True
            return FUNCTIONS.select_point("select", marine_xy)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # Find and move to the nearest mineral.
            minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                      if unit.alliance == _PLAYER_NEUTRAL]

            if self._previous_mineral_xy in minerals:
                # Don't go for the same mineral shard as other marine.
                minerals.remove(self._previous_mineral_xy)

            if minerals:
                # Find the closest.
                distances = numpy.linalg.norm(
                    numpy.array(minerals) - numpy.array(marine_xy), axis=1)
                closest_mineral_xy = minerals[numpy.argmin(distances)]

                # Swap to the other marine.
                self._marine_selected = False
                self._previous_mineral_xy = closest_mineral_xy
                return FUNCTIONS.Move_screen("now", closest_mineral_xy)

        return FUNCTIONS.no_op()