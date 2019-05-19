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



class BuildMarines(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec):
        super(BuildMarines, self).setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent requires the feature_units observation.")

    def reset(self):
        super(BuildMarines, self).reset()

        self.mineral_position = []
        self.potential_supply_positions = []
        self.potential_barrack_positions = []

        self._minerals_or_geyser_index = 0

        self.scv_selected = False
        self.current_supply_number = 0
        self.current_scv_number = 0
        self.current_barrack_number = 0
        self.supply_cnt = 15

        self.build_new_commend = False
        self.commend_center_point = None

    def _get_minerals_position(self):
        self._minerals_or_geyser_index += 1
        if self._minerals_or_geyser_index >= len(self.mineral_position):
            self._minerals_or_geyser_index = 0
        return self.mineral_position[self._minerals_or_geyser_index]

    def step(self, obs):

        # time.sleep(0.01)

        # print(1)

        if self.commend_center_point is None:
            self.commend_center_point = _xy_command_center_point(obs)

        # 在第一次进入游戏的时候，找到合适放补给站和兵营的位置
        if not self.potential_supply_positions:
            self.potential_supply_positions = _get_potential_supply_positions(obs)
        if not self.potential_barrack_positions:
            self.potential_barrack_positions = _get_potential_barrack_positions(obs)
        # 找到所有矿的位置
        if not self.mineral_position:
            self.mineral_position = _get_minerals_positions(obs)
            # print("***", self._minerals_or_geyser_positions)

        self.supply_cnt = cal_food(obs)
        self.current_supply_number = get_num(obs, units.Terran.SupplyDepot)
        self.current_barrack_number = get_num(obs, units.Terran.Barracks)
        self.current_scv_number = get_num(obs, units.Terran.SCV)
        self.current_CommandCenter_number = get_num(obs, units.Terran.CommandCenter)

        # print('begin')
        # 有空闲SCV，就给SCV安排任务
        idle_scv_pos = get_idle_position(obs, units.Terran.SCV) #先找空闲的SCV
        if idle_scv_pos is not None:
            # print('SCV')
            # 当前选中了一个SCV
            if FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions and check_selected(obs, idle_scv_pos):
                import pickle
                pickle.dump(obs, open(r'obs.txt', 'wb'))
                minerals_pos = self._get_minerals_position()
                return FUNCTIONS.Harvest_Gather_screen("now", minerals_pos)
            else:
                return FUNCTIONS.select_point("select", idle_scv_pos)

        # 有空闲兵营，有人口空余，就造兵
        idle_barrack_pos = get_idle_position(obs, units.Terran.Barracks)
        if idle_barrack_pos is not None and obs.observation.player[_FOOD_USED] < obs.observation.player[_FOOD_CAP]:
            # print('build Marine')
            # 已经选中了兵营
            if FUNCTIONS.Train_Marine_quick.id in obs.observation.available_actions and check_selected(obs, idle_barrack_pos):
                # import pickle
                # pickle.dump(obs, open(r'obs.txt', 'wb'))
                if len(obs.observation.build_queue) < 2 and \
                        self.supply_cnt > obs.observation.player[_FOOD_USED] + len(obs.observation.build_queue):
                    return FUNCTIONS.Train_Marine_quick('now')
            else:
                return FUNCTIONS.select_point("select", idle_barrack_pos)

        # 有空闲基地，有人口，SCV不够多，有钱造新的SCV
        idle_commandcenter_pos = get_idle_position(obs, units.Terran.CommandCenter)
        # print(idle_commandcenter_pos, self.current_scv_number, len(self.mineral_position), obs.observation.player[_FOOD_USED], self.supply_cnt, obs.observation.player[_MINERAL_COLLECTED])
        if idle_commandcenter_pos is not None and self.current_scv_number <= 2.5 * len(self.mineral_position) and \
                obs.observation.player[_FOOD_USED] < self.supply_cnt and obs.observation.player[_MINERAL_COLLECTED] >= 50:
            # 已经选中了基地
            # print('build SCV')
            if FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions and check_selected(obs, idle_commandcenter_pos):
                if self.supply_cnt > obs.observation.player[_FOOD_USED] + len(obs.observation.build_queue) and \
                        self.current_scv_number + len(obs.observation.build_queue) <= 2.5 * len(self.mineral_position):
                    return FUNCTIONS.Train_SCV_quick("now")

            else:
                return FUNCTIONS.select_point("select", idle_commandcenter_pos)


        # 有钱，预支人口不够，还有地方，就造补给站
        if obs.observation.player[_MINERAL_COLLECTED] >= 100 and \
                self.supply_cnt - self.current_barrack_number * 2 - self.current_CommandCenter_number * 2 <= obs.observation.player[_FOOD_USED] and \
                self.current_supply_number < len(self.potential_supply_positions):
            # print('build Supplyment', self.supply_cnt, self.current_barrack_number, self.current_CommandCenter_number, obs.observation.player[_FOOD_USED])
            if FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions:
                pos = self.potential_supply_positions[self.current_supply_number]
                # print("^^^ Build_SupplyDepot_screen")
                # print('build')
                return FUNCTIONS.Build_SupplyDepot_screen("now", pos)
            else:
                # print('select')
                scv_pos = get_random_position(obs, units.Terran.SCV)
                return FUNCTIONS.select_point("select", scv_pos)

        # 有钱，还有地方，就造兵营
        if obs.observation.player[_MINERAL_COLLECTED] >= 150 and self.current_barrack_number < len(self.potential_barrack_positions):
            # print('build Barrack')
            if FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:
                # print('build')
                pos = self.potential_barrack_positions[self.current_barrack_number]
                return FUNCTIONS.Build_Barracks_screen('now', pos)
            else:
                # print('select')
                scv_pos = get_random_position(obs, units.Terran.SCV)
                return FUNCTIONS.select_point("select", scv_pos)




        # print('no op')
        return FUNCTIONS.no_op()