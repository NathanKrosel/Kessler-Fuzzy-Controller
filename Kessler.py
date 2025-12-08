# References:
# https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_tipping_problem.html
# https://www.kaggle.com/code/emineyetm/creating-and-plotting-triangular-fuzzy-membership
# ECE 449 Lab 5, Lab 4
# https://github.com/danielwilczak101/EasyGA
# https://github.com/ThalesGroup/kessler-game/tree/main

import math
import random
import time
from typing import Dict, Tuple

import EasyGA
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from kesslergame import GraphicsType, Scenario, TrainerEnvironment, KesslerController  # type: ignore

# --- CustomController Class (GA Optimized) ---

class CustomController(KesslerController):

    def __init__(self, chromosome=None):
        self.eval_frames = 0
        self.last_mine_frame = -1000  # Track when we last dropped a mine

        # =================================================================
        # 1. Targeting Control (STATIC - Not GA Optimized in this version)
        # =================================================================

        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians
        ship_turn = ctrl.Consequent(np.arange(-180,180,3), 'ship_turn') # Degrees
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')

        # Declare fuzzy sets for bullet_time
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        # Declare fuzzy sets for theta_delta
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

        # Declare fuzzy sets for the ship_turn consequent
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        #Declare singleton fuzzy sets for the ship_fire consequent
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

        #Declare each fuzzy rule (Rules 4, 11, 18 removed as in original)
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        self.targeting_control = ctrl.ControlSystem([
            rule1, rule2, rule3, rule5, rule6, rule7, rule8, rule9, rule10,
            rule12, rule13, rule14, rule15, rule16, rule17, rule19, rule20, rule21
        ])

        # =================================================================
        # 2. Thrust Control (GA OPTIMIZED)
        # =================================================================
        asteroid_distance = ctrl.Antecedent(np.arange(0, 1000, 10), 'asteroid_distance')
        asteroid_vel = ctrl.Antecedent(np.arange(0, 200, 5), 'asteroid_vel')
        theta_diff = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.1), 'theta_diff')
        ship_thrust = ctrl.Consequent(np.arange(-1.0, 1.0, 0.05), 'ship_thrust')

        # =================================================================
        # 3. Mine Control (GA OPTIMIZED) - FIXED!
        # =================================================================
        mine_distance = ctrl.Antecedent(np.arange(0, 300, 10), 'mine_distance')
        mine_asteroid_vel = ctrl.Antecedent(np.arange(0, 200, 5), 'mine_asteroid_vel')
        mine_alignment = ctrl.Antecedent(np.arange(0, math.pi, 0.1), 'mine_alignment')  # FIXED: Now 0 to π
        mine_deploy = ctrl.Consequent(np.arange(-1, 1, 0.1), 'mine_deploy')

        # --- GA-Optimization: Parsing the Chromosome (54 Genes Total) ---
        default_full = np.zeros(54)
        # asteroid_distance (8 genes)
        default_full[0:8] = [0.000007428519362, 0.100082374621843, 0.199985372946384, 0.175018293956471, 0.400019274586219, 0.750013628914753, 0.599982184957362, 0.999998527384916]
        # asteroid_vel (7 genes)
        default_full[8:15] = [0.000003918274635, 0.049985284739162, 0.110008392746158, 0.075007362849517, 0.150015297463829, 0.225002271849537, 0.149998736291847]
        # theta_diff (13 genes)
        default_full[15:28] = [0.330004756192837, 0.499998183726495, 0.670007293847518, 0.669998374951826, 0.750001829364758, 0.829999274685193, 0.170016839284756, 0.250002478391562, 0.329998184756283, 0.000006283947518, 0.167000573829461, 0.829998184756293, 0.999999284756193]
        # ship_thrust (9 genes)
        default_full[28:37] = [0.000005182736491, 0.100009728465193, 0.299997028475619, 0.200001983746251, 0.500003928475162, 0.699999284756391, 0.749997482938475, 0.900001847562938, 0.999999647382951]
        # mine_distance (6 genes) - ADJUSTED DEFAULTS
        default_full[37:43] = [0.000008374629517, 0.099998284756391, 0.200001983746251, 0.299998018475629, 0.500002972837465, 0.800000392847516]
        # mine_asteroid_vel (6 genes)
        default_full[43:49] = [0.000004928374651, 0.199999201847562, 0.400002938475162, 0.499998374625183, 0.700000284756193, 0.999999183746582]
        # mine_alignment (4 genes) - ADJUSTED DEFAULTS
        default_full[49:53] = [0.100002847563918, 0.249999374625183, 0.599998284756193, 0.799999184756293]
        # mine_deploy threshold (1 gene) - LOWERED DEFAULT
        default_full[53] = 0.200000374625183

        genes = np.array(chromosome, dtype=float) if chromosome is not None else None

        if genes is None:
            print("WARNING: Using default fuzzy set parameters. No chromosome provided.")
            genes = default_full
        elif len(genes) < 54:
            print("WARNING: Chromosome shorter than 54 genes; padding remaining with defaults.")
            padded = default_full.copy()
            padded[:len(genes)] = genes
            genes = padded
        
        # 2.1 Asteroid Distance (8 genes, scaled 0-1000)
        dist_genes = genes[0:8]
        dist_trimf = dist_genes[0:6].reshape(2, 3)
        dist_trimf = np.array([np.sort(row) for row in dist_trimf]) * 1000
        dist_smf = np.sort(dist_genes[6:8]) * 1000
        asteroid_distance['S'] = fuzz.trimf(asteroid_distance.universe, [dist_trimf[0,0], dist_trimf[0,1], dist_trimf[0,2]])
        asteroid_distance['M'] = fuzz.trimf(asteroid_distance.universe, [dist_trimf[1,0], dist_trimf[1,1], dist_trimf[1,2]])
        asteroid_distance['L'] = fuzz.smf(asteroid_distance.universe, dist_smf[0], dist_smf[1])

        # 2.2 Asteroid Velocity (7 genes, scaled 0-200)
        vel_genes = genes[8:15]
        vel_genes[0:3] = np.sort(vel_genes[0:3])
        vel_genes[3:6] = np.sort(vel_genes[3:6])
        vel_L_start = min(vel_genes[6] * 200, 200)
        asteroid_vel['S'] = fuzz.trimf(asteroid_vel.universe, [vel_genes[0]*200, vel_genes[1]*200, vel_genes[2]*200])
        asteroid_vel['M'] = fuzz.trimf(asteroid_vel.universe, [vel_genes[3]*200, vel_genes[4]*200, vel_genes[5]*200])
        asteroid_vel['L'] = fuzz.smf(asteroid_vel.universe, vel_L_start, 200)

        # 2.3 Theta Diff (13 genes, scaled -pi to pi)
        theta_diff_trimf = genes[15:24].reshape(3, 3)
        theta_diff_trimf = np.array([np.sort(row) for row in theta_diff_trimf])
        theta_diff_smf = genes[24:28].reshape(2, 2)
        theta_diff_smf = np.array([np.sort(row) for row in theta_diff_smf])
        theta_diff['S'] = fuzz.trimf(theta_diff.universe, [self.scale_theta(theta_diff_trimf[0,0]), self.scale_theta(theta_diff_trimf[0,1]), self.scale_theta(theta_diff_trimf[0,2])])
        theta_diff['PM'] = fuzz.trimf(theta_diff.universe, [self.scale_theta(theta_diff_trimf[1,0]), self.scale_theta(theta_diff_trimf[1,1]), self.scale_theta(theta_diff_trimf[1,2])])
        theta_diff['NM'] = fuzz.trimf(theta_diff.universe, [self.scale_theta(theta_diff_trimf[2,0]), self.scale_theta(theta_diff_trimf[2,1]), self.scale_theta(theta_diff_trimf[2,2])])
        
        theta_diff['NL'] = fuzz.zmf(theta_diff.universe, self.scale_theta(min(theta_diff_smf[0,0], theta_diff_smf[0,1])), self.scale_theta(max(theta_diff_smf[0,0], theta_diff_smf[0,1])))
        theta_diff['PL'] = fuzz.smf(theta_diff.universe,  self.scale_theta(min(theta_diff_smf[1,0], theta_diff_smf[1,1])), self.scale_theta(max(theta_diff_smf[1,0], theta_diff_smf[1,1])))

        # 2.4 Ship Thrust (9 genes, scaled -1.0 to 1.0)
        thrust_genes_mag = np.sort(genes[28:37].reshape(3, 3), axis=1)
        
        # Back Thrust (BH, BM, BL)
        ship_thrust['BH'] = fuzz.trimf(ship_thrust.universe, [-thrust_genes_mag[2,2], -thrust_genes_mag[2,1], -thrust_genes_mag[2,0]])
        ship_thrust['BM'] = fuzz.trimf(ship_thrust.universe, [-thrust_genes_mag[1,2], -thrust_genes_mag[1,1], -thrust_genes_mag[1,0]])
        ship_thrust['BL'] = fuzz.trimf(ship_thrust.universe, [-thrust_genes_mag[0,2], -thrust_genes_mag[0,1], -thrust_genes_mag[0,0]])
        
        # Low/Zero Thrust (L)
        ship_thrust['L'] = fuzz.trimf(ship_thrust.universe, [-thrust_genes_mag[0,2]*0.3, 0, thrust_genes_mag[0,2]*0.3])
        
        # Forward Thrust (FL, FM, FH)
        ship_thrust['FL'] = fuzz.trimf(ship_thrust.universe, [thrust_genes_mag[0,0], thrust_genes_mag[0,1], thrust_genes_mag[0,2]])
        ship_thrust['FM'] = fuzz.trimf(ship_thrust.universe, [thrust_genes_mag[1,0], thrust_genes_mag[1,1], thrust_genes_mag[1,2]])
        ship_thrust['FH'] = fuzz.trimf(ship_thrust.universe, [thrust_genes_mag[2,0], thrust_genes_mag[2,1], thrust_genes_mag[2,2]])

        # Thrust Control Rules
        rule1_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['L'] & theta_diff['S'], ship_thrust['BH'])
        rule2_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['M'] & theta_diff['S'], ship_thrust['BM'])
        rule3_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['S'] & theta_diff['S'], ship_thrust['BM'])
        rule4_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['L'] & theta_diff['S'], ship_thrust['BM'])
        rule5_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['M'] & theta_diff['S'], ship_thrust['BM'])
        rule6_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['S'] & theta_diff['S'], ship_thrust['BL'])
        rule7_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['L'] & theta_diff['S'], ship_thrust['BM'])
        rule8_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['M'] & theta_diff['S'], ship_thrust['BL'])
        rule9_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['S'] & theta_diff['S'], ship_thrust['BL'])
        rule10_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['L'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FH'])
        rule11_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['M'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FM'])
        rule12_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['S'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FM'])
        rule13_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['L'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FM'])
        rule14_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['M'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FM'])
        rule15_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['S'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FL'])
        rule16_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['L'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FM'])
        rule17_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['M'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FL'])
        rule18_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['S'] & (theta_diff['NL'] | theta_diff['PL']), ship_thrust['FL'])
        rule19_thrust = ctrl.Rule(theta_diff['PM'], ship_thrust['BL'])
        rule20_thrust = ctrl.Rule(theta_diff['NM'], ship_thrust['BL'])

        self.thrust_movement = ctrl.ControlSystem([
            rule1_thrust, rule2_thrust, rule3_thrust, rule4_thrust, rule5_thrust,
            rule6_thrust, rule7_thrust, rule8_thrust, rule9_thrust, rule10_thrust,
            rule11_thrust, rule12_thrust, rule13_thrust, rule14_thrust, rule15_thrust,
            rule16_thrust, rule17_thrust, rule18_thrust, rule19_thrust, rule20_thrust
        ])

        # =================================================================
        # 3. Mine Control System 
        # =================================================================
        
        # 3.1 Mine Distance (6 genes, scaled 0-300)
        mine_dist_genes = genes[37:43]
        mine_dist_trimf = mine_dist_genes[0:6].reshape(2, 3)
        mine_dist_trimf = np.array([np.sort(row) for row in mine_dist_trimf]) * 300
        mine_distance['Close'] = fuzz.trimf(mine_distance.universe, [mine_dist_trimf[0,0], mine_dist_trimf[0,1], mine_dist_trimf[0,2]])
        mine_distance['Far'] = fuzz.trimf(mine_distance.universe, [mine_dist_trimf[1,0], mine_dist_trimf[1,1], mine_dist_trimf[1,2]])

        # 3.2 Mine Asteroid Velocity (6 genes, scaled 0-200)
        mine_vel_genes = genes[43:49]
        mine_vel_trimf = mine_vel_genes[0:6].reshape(2, 3)
        mine_vel_trimf = np.array([np.sort(row) for row in mine_vel_trimf]) * 200
        mine_asteroid_vel['Slow'] = fuzz.trimf(mine_asteroid_vel.universe, [mine_vel_trimf[0,0], mine_vel_trimf[0,1], mine_vel_trimf[0,2]])
        mine_asteroid_vel['Fast'] = fuzz.trimf(mine_asteroid_vel.universe, [mine_vel_trimf[1,0], mine_vel_trimf[1,1], mine_vel_trimf[1,2]])

        # 3.3 Mine Alignment (4 genes, scaled 0 to π) - FIXED!
        mine_align_genes = genes[49:53]
        mine_align_trimf = mine_align_genes[0:4].reshape(2, 2)
        mine_align_trimf = np.array([np.sort(row) for row in mine_align_trimf])
        # Small angle = well aligned (approaching), large angle = not aligned
        mine_alignment['Aligned'] = fuzz.zmf(mine_alignment.universe, mine_align_trimf[0,0] * math.pi, mine_align_trimf[0,1] * math.pi)
        mine_alignment['NotAligned'] = fuzz.smf(mine_alignment.universe, mine_align_trimf[1,0] * math.pi, mine_align_trimf[1,1] * math.pi)

        # 3.4 Mine Deploy Output
        mine_deploy['No'] = fuzz.trimf(mine_deploy.universe, [-1, -1, 0])
        mine_deploy['Yes'] = fuzz.trimf(mine_deploy.universe, [0, 1, 1])

        # Store mine threshold
        self.mine_threshold = genes[53]
        self.mine_threshold = float(np.clip(self.mine_threshold, -1.0, 1.0))

        # Mine Control Rules
        rule1_mine = ctrl.Rule(mine_distance['Close'] & mine_asteroid_vel['Fast'] & mine_alignment['Aligned'], mine_deploy['Yes'])
        rule2_mine = ctrl.Rule(mine_distance['Close'] & mine_asteroid_vel['Slow'] & mine_alignment['Aligned'], mine_deploy['Yes'])
        rule3_mine = ctrl.Rule(mine_distance['Close'] & mine_alignment['NotAligned'], mine_deploy['No'])
        rule4_mine = ctrl.Rule(mine_distance['Far'], mine_deploy['No'])
        rule5_mine = ctrl.Rule(mine_distance['Close'] & mine_asteroid_vel['Fast'] & mine_alignment['NotAligned'], mine_deploy['No'])

        self.mine_control = ctrl.ControlSystem([
            rule1_mine, rule2_mine, rule3_mine, rule4_mine, rule5_mine
        ])

    @staticmethod
    def scale_theta(val):
        # Scales a 0-1 gene value to the -pi to pi range
        return (val * 2*math.pi) - math.pi

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        
        # --- Targeting Logic ---
        closest_asteroid = None
        for a in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None or closest_asteroid["dist"] > curr_dist:
                closest_asteroid = dict(aster = a, dist = curr_dist)
        
        if closest_asteroid is None:
             return 0.0, 0.0, False, False

        # Intercept Calculation
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0])
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800

        # Quadratic formula determinant
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
        if targ_det < 0:
            bullet_t = closest_asteroid["dist"] / bullet_speed
        else:
            intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
            intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

            if intrcpt1 > intrcpt2:
                bullet_t = intrcpt2 if intrcpt2 >= 0 else intrcpt1
            else:
                bullet_t = intrcpt1 if intrcpt1 >= 0 else intrcpt2
            
            if bullet_t < 0:
                bullet_t = closest_asteroid["dist"] / bullet_speed

        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Run Targeting Control
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.compute()
        turn_rate = shooting.output['ship_turn'] * 3

        fire = shooting.output['ship_fire'] >= 0


        # --- Thrust Logic (GA Optimized) ---
        x_diff = closest_asteroid["aster"]["position"][0] - ship_pos_x
        y_diff = closest_asteroid["aster"]["position"][1] - ship_pos_y
        ship_to_asteroid_theta = math.atan2(y_diff, x_diff)
        theta_diff_input = ship_to_asteroid_theta - ((math.pi/180)*ship_state["heading"])
        theta_diff_input = (theta_diff_input + math.pi) % (2 * math.pi) - math.pi

        asteroid_distance = closest_asteroid["dist"]
        
        # Run Thrust Control
        movement = ctrl.ControlSystemSimulation(self.thrust_movement, flush_after_run=1)
        movement.input['asteroid_distance'] = asteroid_distance
        movement.input['asteroid_vel'] = asteroid_vel
        movement.input['theta_diff'] = theta_diff_input
        try:
            movement.compute()
            thrust_raw = movement.output.get('ship_thrust', 0.0)
        except Exception:
            thrust_raw = 0.0
        if np.isnan(thrust_raw):
            thrust_raw = 0.0

        thrust = thrust_raw * 250


        # --- Mine Logic (FIXED!) ---
        drop_mine = False
        try:
            can_deploy = ship_state["can_deploy_mine"]
        except (KeyError, TypeError):
            can_deploy = False
        try:
            mines_remaining = ship_state["mines_remaining"]
        except (KeyError, TypeError):
            mines_remaining = 0
        frames_since_last_mine = self.eval_frames - self.last_mine_frame
        mine_cooldown_ready = frames_since_last_mine > 90
        
        if can_deploy and mines_remaining > 0 and mine_cooldown_ready:
            # Calculate alignment angle using the angle between asteroid velocity and the vector from asteroid to ship
            # Smaller angle = asteroid moving toward ship = better alignment
            asteroid_to_ship_vec = (-x_diff, -y_diff)  # From asteroid to ship
            asteroid_vel_vec = (closest_asteroid["aster"]["velocity"][0], closest_asteroid["aster"]["velocity"][1])
            
            mag_ats = math.sqrt(asteroid_to_ship_vec[0]**2 + asteroid_to_ship_vec[1]**2)
            mag_vel = math.sqrt(asteroid_vel_vec[0]**2 + asteroid_vel_vec[1]**2)
            
            if mag_ats > 0.01 and mag_vel > 0.01:
                # Calculate angle between asteroid velocity and vector to ship
                dot_product = asteroid_to_ship_vec[0] * asteroid_vel_vec[0] + asteroid_to_ship_vec[1] * asteroid_vel_vec[1]
                cos_angle = dot_product / (mag_ats * mag_vel)
                cos_angle = max(-1, min(1, cos_angle))
                alignment_angle = math.acos(cos_angle)  # This gives 0 to π
                # 0 = perfect alignment (heading directly toward ship)
                # π = moving away from ship
                
                # Run Mine Control System
                mine_sim = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
                mine_sim.input['mine_distance'] = min(closest_asteroid["dist"], 300)
                mine_sim.input['mine_asteroid_vel'] = min(asteroid_vel, 200)
                mine_sim.input['mine_alignment'] = min(alignment_angle, math.pi)
                try:
                    mine_sim.compute()
                    mine_output = mine_sim.output.get('mine_deploy', -1)
                except Exception as e:
                    #print(f"Mine fuzzy error: {e}")
                    mine_output = -1
                if np.isnan(mine_output):
                    mine_output = -1

                # Deploy mine if fuzzy output exceeds threshold
                if mine_output >= self.mine_threshold:
                    drop_mine = True
                    self.last_mine_frame = self.eval_frames
                    #print(f"MINE DROPPED! Distance: {closest_asteroid['dist']:.1f}, Vel: {asteroid_vel:.1f}, Alignment: {math.degrees(alignment_angle):.1f}°, Output: {mine_output:.2f}")
             
        # Escape logic after mine drop
        if frames_since_last_mine < 30:
            escape_heading = math.degrees(math.atan2(-y_diff, -x_diff)) % 360
            heading_diff = (escape_heading - ship_state["heading"] + 540) % 360 - 180
            turn_rate = max(min(heading_diff * 3, 180), -180)
            thrust = 300

        self.eval_frames += 1

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "GA Optimized Controller"


# --- GA Setup and Fitness Function ---

class SafeGAController(CustomController):
    """Wrapper to fail safely if fuzzy controller outputs are missing."""
    def actions(self, ship_state, game_state):
        try:
            return super().actions(ship_state, game_state)
        except KeyError as exc:
            if getattr(exc, "args", None) and exc.args[0] in ['ship_thrust', 'ship_turn', 'ship_fire', 'mine_deploy']:
                return 0.0, 0.0, False, False
            return 0.0, 0.0, False, False
        except Exception:
            return 0.0, 0.0, False, False

def fitness(chromosome):
    chromosome = [gene.value for gene in chromosome]

    my_test_scenario = Scenario(name='Test Scenario',
        num_asteroids=10,
        ship_states=[
            {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
        ],
        map_size=(1000, 800),
        time_limit=20,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False)
    
    game_settings = {
        'perf_tracker': False,
        'prints_on': False,
        'graphics_type': GraphicsType.NoGraphics,
        'realtime_multiplier': 0,
        'graphics_obj': None,
        'frequency': 30
    }

    game = TrainerEnvironment(settings=game_settings)
    controller = SafeGAController(chromosome) 
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[controller])

    asteroids_hit = score.teams[0].asteroids_hit
    accuracy = score.teams[0].accuracy
    deaths = score.teams[0].deaths
    
    survival_bonus = (3 - deaths) * 50
    result = asteroids_hit * 10 + accuracy * 100 + survival_bonus - deaths * 100

    return result

def gene_generation():
    return random.uniform(0, 1)

def main():
    # --- EasyGA Initialization ---
    ga = EasyGA.GA()
    ga.chromosome_length = 54
    ga.population_size = 15
    ga.target_fitness_type = 'max'
    ga.generation_goal = 30
    ga.fitness_function_impl = fitness
    ga.gene_impl = lambda: gene_generation()

    ga.database_name = ""
    print('Starting GA Evolution...')
    print(f'Population: {ga.population_size}, Generations: {ga.generation_goal}, Chromosome Length: {ga.chromosome_length}')
    ga.evolve()
    print('\n=== Evolution Complete ===')
    ga.print_best_chromosome()

    best_genes = None
    try:
        if hasattr(ga, 'best_chromosome'):
            print(f'\nBest Fitness: {ga.best_chromosome.fitness}')
            best_genes = [gene.value for gene in ga.best_chromosome]
        elif hasattr(ga, 'population') and len(ga.population) > 0:
            best = max(ga.population, key=lambda x: x.fitness)
            print(f'\nBest Fitness: {best.fitness}')
            best_genes = [gene.value for gene in best]
    except Exception as e:
        print(f'\nCould not retrieve best fitness: {e}')

    # =================================================================
    # VISUAL DEMONSTRATION WITH BEST CHROMOSOME
    # =================================================================
    if best_genes is not None:
        x = input('\nPress y to run visual demonstration with best evolved controller... ')
        while x.lower().strip() == 'y':

            print('\n' + '='*60)
            print('Running visual demonstration with best evolved controller...')
            print('='*60)
            
            from kesslergame import KesslerGame
            
            demo_scenario = Scenario(
                name='Best Controller Demo',
                num_asteroids=10,
                ship_states=[
                    {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                ],
                map_size=(700, 500),
                time_limit=30,
                ammo_limit_multiplier=0,
                stop_if_no_ammo=False
            )
            
            demo_settings = {
                'perf_tracker': True,
                'graphics_type': GraphicsType.Tkinter,
                'realtime_multiplier': 1,
                'graphics_obj': None,
                'frequency': 30
            }
            
            demo_game = KesslerGame(settings=demo_settings)
            best_controller = CustomController(chromosome=best_genes)
            
            print('\nStarting visual demo...')
            demo_start = time.time()
            demo_score, demo_perf = demo_game.run(scenario=demo_scenario, controllers=[best_controller])
            demo_time = time.time() - demo_start
            
            print('\n' + '='*60)
            print('DEMO RESULTS')
            print('='*60)
            print(f'Scenario eval time: {demo_time:.2f}s')
            print(f'Stop reason: {demo_score.stop_reason}')
            print(f'Asteroids hit: {demo_score.teams[0].asteroids_hit}')
            print(f'Deaths: {demo_score.teams[0].deaths}')
            print(f'Accuracy: {demo_score.teams[0].accuracy:.2%}')
            print(f'Mean eval time: {demo_score.teams[0].mean_eval_time:.6f}s')
            print('='*60)

            x = input('Would you like to rerun the visual demonstration? (y/n) ')
        print('\n=== All Done! ===')
    else:
        print('\nWarning: Could not retrieve best genes for visual demo')
