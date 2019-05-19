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


def get_new_command_center(obs):
    x, y, r = None, None, None
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Terran.CommandCenter:
            x, y = unit.x, unit.y
            r = unit.radius
    if x < 42:
        x = x + r + r
    else:
        x = x - r - r
    return tuple((x, y))


def _get_idle_scv_position(obs):
    for unit in obs.observation.feature_units:
        # print(int(unit.order_length) == 0)
        if unit.unit_type == units.Terran.SCV and int(unit.order_length) == 0:
            return tuple((unit.x, unit.y))
    return None


def _get_potential_supply_positions(obs):
    res = []
    unit_type = obs.observation.feature_screen.unit_type
    # 上半部分地图，在上边两个气矿的中间放3个气矿
    mask = unit_type[:unit_type.shape[0] // 2, ] == _VESPENE_GEYSER
    y, x = mask.nonzero()
    y = int(numpy.max(y))
    x = int(numpy.mean(x))
    res.extend([(x, y), (x + 10, y), (x - 10, y)])
    unit_type = obs.observation.feature_screen.unit_type
    mask = unit_type[unit_type.shape[0] // 2:, ] == _VESPENE_GEYSER
    y, x = mask.nonzero()
    y = int(numpy.min(y))
    x = int(numpy.mean(x))
    res.extend([(x, y), (x + 10, y), (x - 10, y)])
    return res


def _get_geyser_positions(obs):
    res = []
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Neutral.VespeneGeyser:
            res.append(tuple((unit.x, unit.y)))
    commend_center_point = _xy_command_center_point(obs)
    res = sorted(res, key=lambda k: numpy.linalg.norm(numpy.array(commend_center_point) - numpy.array(k)))
    return res


def _get_minerals_positions(obs):
    res = []
    for unit in obs.observation.feature_units:
        if unit.unit_type == units.Neutral.MineralField:
            res.append(tuple((unit.x, unit.y)))
    # commend_center_point = _xy_command_center_point(obs)
    # res = sorted(res, key=lambda k: numpy.linalg.norm(numpy.array(commend_center_point) - numpy.array(k)))
    return res


class CollectMineralsAndGas(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec):
        super(CollectMineralsAndGas, self).setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent requires the feature_units observation.")

    def reset(self):
        super(CollectMineralsAndGas, self).reset()
        self.supply_cnt = 0
        self.preserve_supply = 2
        self._minerals_or_geyser_positions = []
        self._minerals_or_geyser_index = 0
        self._scv_selected = False
        self._scv_selected_xy = [0, 0]
        self.current_supply_number = 0
        self.max_geyser_number = 6
        self.build_geyser_min_scv = [12, 12, 12, 26, 29, 32]
        self.potential_supply_positions = []
        self.mineral_used = 0
        self.current_geyser_number = 0
        self.geyser_positions = []
        self._commend_center_selected = False
        self.build_new_commend = False
        self.commend_center_point = None

    def _get_minerals_or_geyser_position(self):
        for i in range(self.current_geyser_number):
            if self.geyser_positions[i] not in self._minerals_or_geyser_positions:
                self._minerals_or_geyser_positions.append(self.geyser_positions[i])
        self._minerals_or_geyser_index += 1
        if self._minerals_or_geyser_index >= len(self._minerals_or_geyser_positions):
            self._minerals_or_geyser_index = 0
        return self._minerals_or_geyser_positions[self._minerals_or_geyser_index]

    def step(self, obs):
        # import pickle
        # pickle.dump(obs, open(r"obs.txt", "wb"))

        time.sleep(0.01)

        if self.commend_center_point is None:
            self.commend_center_point = _xy_command_center_point(obs)
        # 在第一次进入游戏的时候，找到6个合适放补给站的位置
        if not self.potential_supply_positions:
            self.potential_supply_positions = _get_potential_supply_positions(obs)
            # print("*** ", self.potential_supply_positions)
        if not self.geyser_positions:
            self.geyser_positions = _get_geyser_positions(obs)
            # print("$$$ ", self.geyser_positions)
        if not self._minerals_or_geyser_positions:
            self._minerals_or_geyser_positions = _get_minerals_positions(obs)
            # print("***", self._minerals_or_geyser_positions)
        if self.supply_cnt == 0:
            self.supply_cnt = obs.observation.player[_FOOD_CAP]

        # print('SCV selected:', self._scv_selected)

        # 建造新的基地
        # if obs.observation.player[_MINERAL_COLLECTED] - self.mineral_used >= 450 and FUNCTIONS.Build_CommandCenter_screen.id in obs.observation.available_actions:
        if self.build_new_commend is False and FUNCTIONS.Build_CommandCenter_screen.id in obs.observation.available_actions:
            new_command_center = get_new_command_center(obs)
            # print('?????????????', obs.observation.player[_MINERAL_COLLECTED], new_command_center)
            self.build_new_commend = True
            self.supply_cnt += 15
            self.preserve_supply += 2
            return FUNCTIONS.Build_CommandCenter_screen("now", new_command_center)

        # 人数已达上限，造补给站
        if obs.observation.player[_MINERAL_COLLECTED] >= 100 and self.supply_cnt - self.preserve_supply <= obs.observation.player[
            _FOOD_USED] and FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions and self.current_supply_number < 3:
            pos = self.potential_supply_positions[self.current_supply_number]
            self.current_supply_number += 1
            self.mineral_used += 100
            self.supply_cnt += 8
            # print("^^^ Build_SupplyDepot_screen")
            return FUNCTIONS.Build_SupplyDepot_screen("now", pos)

        # todo： now 和 queue的区别
        # 当前人数超过一定数目，就开矿
        # if obs.observation.player[_MINERAL_COLLECTED] >= 75 and obs.observation.player[
        #     _FOOD_USED] > self.build_geyser_min_scv[
        #     self.current_geyser_number] and FUNCTIONS.Build_Refinery_screen.id in obs.observation.available_actions and self.current_geyser_number < 2:
        #     pos = self.geyser_positions[self.current_geyser_number]
        #     self.current_geyser_number += 1
        #     self.mineral_used += 75
        #     # print("^^^ Build_Refinery_screen")
        #     return FUNCTIONS.Build_Refinery_screen("now", pos)

        # 给人安排任务
        idle_scv_pos = _get_idle_scv_position(obs)
        if idle_scv_pos is None:

            # print("every body is busy")
            if self._commend_center_selected is False and obs.observation.player[_MINERAL_COLLECTED] >= 50:

                self._commend_center_selected = True
                self._scv_selected = False
                commend_center_point = _xy_command_center_point(obs)
                return FUNCTIONS.select_point("select", commend_center_point)

            if self._commend_center_selected is True and FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions \
                    and self.supply_cnt > obs.observation.player[_FOOD_USED] + len(obs.observation.build_queue):
                self.mineral_used += 50
                self._commend_center_selected = False
                # print("^^^ Train_SCV_quick")
                return FUNCTIONS.Train_SCV_quick("now")

            return FUNCTIONS.no_op()

        # print(idle_scv_pos, self._scv_selected)
        if self._scv_selected is False:
            # print('select SCV')
            self._scv_selected = True
            self._commend_center_selected = False
            self._scv_selected_xy = idle_scv_pos
            return FUNCTIONS.select_point("select", self._scv_selected_xy)
        elif self._scv_selected is True:
            self._scv_selected = False
            # print('SCV go to work!')
            # 轮流选矿
            # print(obs.observation.player[_MINERAL_COLLECTED], obs.observation.score_cumulative.score)
            if obs.observation.score_cumulative.score > 700:
                _minerals_or_geyser_pos = self._get_minerals_or_geyser_position()
                return FUNCTIONS.Harvest_Gather_screen("now", _minerals_or_geyser_pos)
            else:
                mineral_fields = [unit for unit in obs.observation.feature_units
                                  if unit.unit_type == units.Neutral.MineralField]
                # 选离工人最近的矿
                mineral_fields_xy = [[unit.x, unit.y] for unit in mineral_fields]
                # 选离基地最近的矿
                # distances = numpy.linalg.norm(numpy.array(mineral_fields_xy) - numpy.array(self._scv_selected_xy), axis=1)
                distances = numpy.linalg.norm(numpy.array(mineral_fields_xy) - numpy.array(self.commend_center_point),
                                              axis=1)
                closest_mineral = mineral_fields[numpy.argmin(distances)]
                return FUNCTIONS.Harvest_Gather_screen("now", [closest_mineral.x, closest_mineral.y])

        return FUNCTIONS.no_op()
