from typing import Dict, Tuple, Optional
import math

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from kesslergame import KesslerController


class ProjectController(KesslerController):
    def __init__(self):
        self.eval_frames = 0

        # Targeting fuzzy controller (based on ScottDickController)
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), "bullet_time")
        theta_delta = ctrl.Antecedent(
            np.arange(-1 * math.pi / 30, math.pi / 30, 0.1),
            "theta_delta",
        )  # radians

        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), "ship_turn")  # deg
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), "ship_fire")

        bullet_time["S"] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time["M"] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time["L"] = fuzz.smf(bullet_time.universe, 0.0, 0.1)

        theta_delta["NL"] = fuzz.zmf(
            theta_delta.universe,
            -1 * math.pi / 30,
            -2 * math.pi / 90,
        )
        theta_delta["NM"] = fuzz.trimf(
            theta_delta.universe,
            [-1 * math.pi / 30, -2 * math.pi / 90, -1 * math.pi / 90],
        )
        theta_delta["NS"] = fuzz.trimf(
            theta_delta.universe,
            [-2 * math.pi / 90, -1 * math.pi / 90, math.pi / 90],
        )
        theta_delta["PS"] = fuzz.trimf(
            theta_delta.universe,
            [-1 * math.pi / 90, math.pi / 90, 2 * math.pi / 90],
        )
        theta_delta["PM"] = fuzz.trimf(
            theta_delta.universe,
            [math.pi / 90, 2 * math.pi / 90, math.pi / 30],
        )
        theta_delta["PL"] = fuzz.smf(
            theta_delta.universe,
            2 * math.pi / 90,
            math.pi / 30,
        )

        ship_turn["NL"] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn["NM"] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn["NS"] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
        ship_turn["PS"] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn["PM"] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn["PL"] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        ship_fire["N"] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire["Y"] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        rule1 = ctrl.Rule(
            bullet_time["L"] & theta_delta["NL"],
            (ship_turn["NL"], ship_fire["N"]),
        )
        rule2 = ctrl.Rule(
            bullet_time["L"] & theta_delta["NM"],
            (ship_turn["NM"], ship_fire["N"]),
        )
        rule3 = ctrl.Rule(
            bullet_time["L"] & theta_delta["NS"],
            (ship_turn["NS"], ship_fire["Y"]),
        )
        rule5 = ctrl.Rule(
            bullet_time["L"] & theta_delta["PS"],
            (ship_turn["PS"], ship_fire["Y"]),
        )
        rule6 = ctrl.Rule(
            bullet_time["L"] & theta_delta["PM"],
            (ship_turn["PM"], ship_fire["N"]),
        )
        rule7 = ctrl.Rule(
            bullet_time["L"] & theta_delta["PL"],
            (ship_turn["PL"], ship_fire["N"]),
        )

        rule8 = ctrl.Rule(
            bullet_time["M"] & theta_delta["NL"],
            (ship_turn["NL"], ship_fire["N"]),
        )
        rule9 = ctrl.Rule(
            bullet_time["M"] & theta_delta["NM"],
            (ship_turn["NM"], ship_fire["N"]),
        )
        rule10 = ctrl.Rule(
            bullet_time["M"] & theta_delta["NS"],
            (ship_turn["NS"], ship_fire["Y"]),
        )
        rule12 = ctrl.Rule(
            bullet_time["M"] & theta_delta["PS"],
            (ship_turn["PS"], ship_fire["Y"]),
        )
        rule13 = ctrl.Rule(
            bullet_time["M"] & theta_delta["PM"],
            (ship_turn["PM"], ship_fire["N"]),
        )
        rule14 = ctrl.Rule(
            bullet_time["M"] & theta_delta["PL"],
            (ship_turn["PL"], ship_fire["N"]),
        )

        rule15 = ctrl.Rule(
            bullet_time["S"] & theta_delta["NL"],
            (ship_turn["NL"], ship_fire["Y"]),
        )
        rule16 = ctrl.Rule(
            bullet_time["S"] & theta_delta["NM"],
            (ship_turn["NM"], ship_fire["Y"]),
        )
        rule17 = ctrl.Rule(
            bullet_time["S"] & theta_delta["NS"],
            (ship_turn["NS"], ship_fire["Y"]),
        )
        rule19 = ctrl.Rule(
            bullet_time["S"] & theta_delta["PS"],
            (ship_turn["PS"], ship_fire["Y"]),
        )
        rule20 = ctrl.Rule(
            bullet_time["S"] & theta_delta["PM"],
            (ship_turn["PM"], ship_fire["Y"]),
        )
        rule21 = ctrl.Rule(
            bullet_time["S"] & theta_delta["PL"],
            (ship_turn["PL"], ship_fire["Y"]),
        )

        self.targeting_control = ctrl.ControlSystem()
        for r in [
            rule1,
            rule2,
            rule3,
            rule5,
            rule6,
            rule7,
            rule8,
            rule9,
            rule10,
            rule12,
            rule13,
            rule14,
            rule15,
            rule16,
            rule17,
            rule19,
            rule20,
            rule21,
        ]:
            self.targeting_control.addrule(r)

        # Navigation & mining fuzzy controller
        danger_level = ctrl.Antecedent(np.arange(0, 1.01, 0.05), "danger")
        big_near = ctrl.Antecedent(np.arange(0, 1.01, 0.05), "big_near")

        thrust_level = ctrl.Consequent(np.arange(0, 201, 5), "thrust")
        mine_decision = ctrl.Consequent(np.arange(-1, 1.01, 0.1), "mine")

        danger_level["L"] = fuzz.trimf(danger_level.universe, [0.0, 0.0, 0.3])
        danger_level["M"] = fuzz.trimf(danger_level.universe, [0.1, 0.5, 0.8])
        danger_level["H"] = fuzz.trimf(danger_level.universe, [0.6, 1.0, 1.0])

        big_near["N"] = fuzz.trimf(big_near.universe, [0.0, 0.0, 0.4])
        big_near["Y"] = fuzz.trimf(big_near.universe, [0.3, 1.0, 1.0])

        thrust_level["STOP"] = fuzz.trimf(thrust_level.universe, [0, 0, 40])
        thrust_level["CRUISE"] = fuzz.trimf(thrust_level.universe, [20, 80, 140])
        thrust_level["BOOST"] = fuzz.trimf(thrust_level.universe, [100, 200, 200])

        mine_decision["N"] = fuzz.trimf(mine_decision.universe, [-1, -1, 0.0])
        mine_decision["Y"] = fuzz.trimf(mine_decision.universe, [0.0, 1, 1])

        nav_rules = [
            ctrl.Rule(
                danger_level["L"] & big_near["N"],
                (thrust_level["CRUISE"], mine_decision["N"]),
            ),
            ctrl.Rule(
                danger_level["M"] & big_near["N"],
                (thrust_level["CRUISE"], mine_decision["N"]),
            ),
            ctrl.Rule(
                danger_level["H"] & big_near["N"],
                (thrust_level["BOOST"], mine_decision["N"]),
            ),
            ctrl.Rule(
                danger_level["L"] & big_near["Y"],
                (thrust_level["CRUISE"], mine_decision["Y"]),
            ),
            ctrl.Rule(
                danger_level["M"] & big_near["Y"],
                (thrust_level["CRUISE"], mine_decision["Y"]),
            ),
            ctrl.Rule(
                danger_level["H"] & big_near["Y"],
                (thrust_level["BOOST"], mine_decision["Y"]),
            ),
        ]

        self.navigation_control = ctrl.ControlSystem()
        for r in nav_rules:
            self.navigation_control.addrule(r)

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _nearest_asteroid(ship_state: Dict, game_state: Dict) -> Optional[Dict]:
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        closest: Optional[Dict] = None
        for a in game_state["asteroids"]:
            curr_dist = math.sqrt(
                (ship_pos_x - a["position"][0]) ** 2
                + (ship_pos_y - a["position"][1]) ** 2
            )
            if closest is None or curr_dist < closest["dist"]:
                closest = {"aster": a, "dist": curr_dist}
        return closest

    @staticmethod
    def _nearest_big_asteroid(ship_state: Dict, game_state: Dict) -> Optional[Dict]:
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        closest: Optional[Dict] = None
        for a in game_state["asteroids"]:
            if a.get("size", 1) <= 1:
                continue
            curr_dist = math.sqrt(
                (ship_pos_x - a["position"][0]) ** 2
                + (ship_pos_y - a["position"][1]) ** 2
            )
            if closest is None or curr_dist < closest["dist"]:
                closest = {"aster": a, "dist": curr_dist}
        return closest

    # ----------------------------------------------------------------------
    # Main control method
    # ----------------------------------------------------------------------
    def actions(
        self, ship_state: Dict, game_state: Dict
    ) -> Tuple[float, float, bool, bool]:
        self.eval_frames += 1

        closest_asteroid = self._nearest_asteroid(ship_state, game_state)
        if closest_asteroid is None:
            return 50.0, 0.0, False, False

        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)

        asteroid_direction = math.atan2(
            closest_asteroid["aster"]["velocity"][1],
            closest_asteroid["aster"]["velocity"][0],
        )
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)

        asteroid_vel = math.sqrt(
            closest_asteroid["aster"]["velocity"][0] ** 2
            + closest_asteroid["aster"]["velocity"][1] ** 2
        )
        bullet_speed = 800

        targ_det = (
            -2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2
        ) ** 2 - (
            4 * (asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"] ** 2)
        )

        if targ_det < 0:
            targ_det = 0.0

        denom = 2 * (asteroid_vel**2 - bullet_speed**2)
        if abs(denom) < 1e-6:
            intrcpt1 = intrcpt2 = 0.0
        else:
            intrcpt1 = (
                (2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)
                + math.sqrt(targ_det)
            ) / denom
            intrcpt2 = (
                (2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)
                - math.sqrt(targ_det)
            ) / denom

        if intrcpt1 > intrcpt2:
            bullet_t = intrcpt2 if intrcpt2 >= 0 else intrcpt1
        else:
            bullet_t = intrcpt1 if intrcpt1 >= 0 else intrcpt2

        intrcpt_x = closest_asteroid["aster"]["position"][0] + (
            closest_asteroid["aster"]["velocity"][0] * (bullet_t + 1 / 30)
        )
        intrcpt_y = closest_asteroid["aster"]["position"][1] + (
            closest_asteroid["aster"]["velocity"][1] * (bullet_t + 1 / 30)
        )

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y), (intrcpt_x - ship_pos_x))

        shooting_theta = my_theta1 - ((math.pi / 180) * ship_state["heading"])
        shooting_theta = self._wrap_angle(shooting_theta)

        shooting = ctrl.ControlSystemSimulation(
            self.targeting_control,
            flush_after_run=1,
        )
        shooting.input["bullet_time"] = float(max(0.0, bullet_t))
        shooting.input["theta_delta"] = float(shooting_theta)
        shooting.compute()

        turn_rate = float(shooting.output["ship_turn"])
        fire = shooting.output["ship_fire"] >= 0.0

        danger_dist_scale = 300.0
        danger_val = max(
            0.0,
            min(
                1.0,
                (danger_dist_scale - closest_asteroid["dist"])
                / danger_dist_scale,
            ),
        )

        big_ast = self._nearest_big_asteroid(ship_state, game_state)
        if big_ast is None:
            big_near_val = 0.0
        else:
            big_near_scale = 350.0
            big_near_val = max(
                0.0,
                min(1.0, (big_near_scale - big_ast["dist"]) / big_near_scale),
            )

        navigation = ctrl.ControlSystemSimulation(
            self.navigation_control,
            flush_after_run=1,
        )
        navigation.input["danger"] = float(danger_val)
        navigation.input["big_near"] = float(big_near_val)
        navigation.compute()

        thrust = float(navigation.output["thrust"])
        drop_mine = navigation.output["mine"] >= 0.0

        # After dropping a mine near a big asteroid, turn and thrust away from it
        if big_ast is not None and big_ast["dist"] < 220:
            drop_mine = True
            ship_heading_rad = ship_state["heading"] * math.pi / 180.0
            vec_x_big = big_ast["aster"]["position"][0] - ship_pos_x
            vec_y_big = big_ast["aster"]["position"][1] - ship_pos_y
            escape_angle = math.atan2(-vec_y_big, -vec_x_big)
            rel_escape = self._wrap_angle(escape_angle - ship_heading_rad)
            if rel_escape > 0:
                turn_rate = 150.0
            else:
                turn_rate = -150.0
            thrust = 200.0

        if closest_asteroid is not None and closest_asteroid["dist"] < 120:
            ship_heading_rad = ship_state["heading"] * math.pi / 180.0
            vec_x = closest_asteroid["aster"]["position"][0] - ship_pos_x
            vec_y = closest_asteroid["aster"]["position"][1] - ship_pos_y
            asteroid_angle = math.atan2(vec_y, vec_x)
            rel_angle = self._wrap_angle(asteroid_angle - ship_heading_rad)

            if abs(rel_angle) < math.pi * 3 / 4:
                if rel_angle > 0:
                    turn_rate = -150.0
                else:
                    turn_rate = 150.0

        thrust = max(0.0, min(thrust, 200.0))
        turn_rate = max(-180.0, min(turn_rate, 180.0))

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Project Controller"


