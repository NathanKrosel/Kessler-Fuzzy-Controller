# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

# References:
# https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_tipping_problem.html
# https://www.kaggle.com/code/emineyetm/creating-and-plotting-triangular-fuzzy-membership
# Lab 5, Lab 4
# https://github.com/danielwilczak101/EasyGA
# https://github.com/ThalesGroup/kessler-game/tree/main


from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt




class CustomController(KesslerController):



    def __init__(self, chromosome=None):
        self.eval_frames = 0 #What is this?
        self.last_mine_frame = -1000  # Track when we last dropped a mine

        # self.targeting_control is the targeting rulebase, which is static in this controller.
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,3), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')

        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()



        # Declare the fuzzy controller, add the rules
        # This is an instance variable, and thus available for other methods in the same object. See notes.
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)


        # controller setup logic for thrust
        asteroid_distance = ctrl.Antecedent(np.arange(0, 1000, 10), 'asteroid_distance') # 1000 as max window size is 1000
        asteroid_vel = ctrl.Antecedent(np.arange(0, 200, 5), 'asteroid_vel') # 200 set bc mostly read values aroind 100
        theta_diff = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.1), 'theta_diff') # normalized range
        ship_thrust = ctrl.Consequent(np.arange(-1.0, 1.0, 0.05), 'ship_thrust')

        # large always has a later range than small
        # Allow GA-provided genes when available, otherwise fall back to defaults.
        genes = np.array(chromosome, dtype=float) if chromosome is not None else None
        if genes is not None and len(genes) >= 15:
            asteroid_distance_genes = genes[0:8]
            asteroid_vel_genes = genes[8:15]
        else:
            asteroid_distance_genes = np.array([0, 0, 0.2, 0.175, 0.400, 0.750, 0.600, 0.1000])
            asteroid_vel_genes = np.array([0, 0, 0.110, 0.075, 0.150, 0.225, 0.150])

        asteroid_distance_trimf = asteroid_distance_genes[0:6] # first 6 genes are for S and M trimf
        asteroid_distance_trimf = asteroid_distance_trimf.reshape(2, 3)
        asteroid_distance_trimf = np.array([np.sort(row) for row in asteroid_distance_trimf])
        asteroid_distance_smf = np.sort(asteroid_distance_genes[6:]) # last 2 genes for L smf
        # scale by 1000
        asteroid_distance['S'] = fuzz.trimf(asteroid_distance.universe, [asteroid_distance_trimf[0,0]*1000, asteroid_distance_trimf[0,1]*1000, asteroid_distance_trimf[0,2]*1000])
        asteroid_distance['M'] = fuzz.trimf(asteroid_distance.universe, [asteroid_distance_trimf[1,0]*1000, asteroid_distance_trimf[1,1]*1000, asteroid_distance_trimf[1,2]*1000])
        asteroid_distance['L'] = fuzz.smf(asteroid_distance.universe, asteroid_distance_smf[0]*1000, asteroid_distance_smf[1]*1000)

        # Velocity genes: 6 for S/M trimf, 1 for L smf start (upper bound fixed to universe max)
        asteroid_vel_genes = asteroid_vel_genes.copy()
        if len(asteroid_vel_genes) < 7:
            # pad if somehow short
            asteroid_vel_genes = np.pad(asteroid_vel_genes, (0, 7 - len(asteroid_vel_genes)), constant_values=0.0)
        asteroid_vel_genes[0:3] = np.sort(asteroid_vel_genes[0:3])
        asteroid_vel_genes[3:6] = np.sort(asteroid_vel_genes[3:6])
        vel_L_start = min(asteroid_vel_genes[6] * 300, 200)
        # scaled by 300
        asteroid_vel['S'] = fuzz.trimf(asteroid_vel.universe, [asteroid_vel_genes[0]*300, asteroid_vel_genes[1]*300, asteroid_vel_genes[2]*300])
        asteroid_vel['M'] = fuzz.trimf(asteroid_vel.universe, [asteroid_vel_genes[3]*300, asteroid_vel_genes[4]*300, asteroid_vel_genes[5]*300])
        asteroid_vel['L'] = fuzz.smf(asteroid_vel.universe, vel_L_start, 200)

        # SIMULATED GENES:
        theta_diff_genes = np.array([0.33, 0.5, 0.67, 0.67, 0.75, 0.83, 0.17, 0.25, 0.33, 0, 0.167, 0.83, 1])
        theta_diff_trimf = theta_diff_genes[0:9] # first 9 genes are for S, PM, NM trimf
        theta_diff_trimf = theta_diff_trimf.reshape(3, 3)
        theta_diff_smf = theta_diff_genes[9:]
        theta_diff_smf = theta_diff_smf.reshape(2,2)
        # small theta diff between ship and closest asteroid (asteroid to the front)
        theta_diff['S'] = fuzz.trimf(theta_diff.universe, [self.scale_theta(theta_diff_trimf[0,0]), self.scale_theta(theta_diff_trimf[0,1]), self.scale_theta(theta_diff_trimf[0,2])])
        # positive medium theta diff (asteroid to the right)
        theta_diff['PM'] = fuzz.trimf(theta_diff.universe, [self.scale_theta(theta_diff_trimf[1,0]), self.scale_theta(theta_diff_trimf[1,1]), self.scale_theta(theta_diff_trimf[1,2])])
        # negative medium theta diff (asteroid to the left)
        theta_diff['NM'] = fuzz.trimf(theta_diff.universe, [self.scale_theta(theta_diff_trimf[2,0]), self.scale_theta(theta_diff_trimf[2,1]), self.scale_theta(theta_diff_trimf[2,2])])
        # large theta diff (asteroid near the back)
        theta_diff['L'] = np.fmax(
                fuzz.zmf(theta_diff.universe, self.scale_theta(theta_diff_smf[0,0]), self.scale_theta(theta_diff_smf[0,1])),
                fuzz.smf(theta_diff.universe,  self.scale_theta(theta_diff_smf[1,0]), self.scale_theta(theta_diff_smf[1,1]))
            )

        # SIMULATED GENES:
        ship_thrust_genes = np.array([0, 0, 0.3, 0.2, 0.5, 0.7, 0.75, 1, 1])
        ship_thrust_genes_trimf = ship_thrust_genes.reshape(3, 3)
        # thrust back at high acceleration
        ship_thrust['BH'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[2,2]*-1, ship_thrust_genes_trimf[2,1]*-1, ship_thrust_genes_trimf[2,0]*-1])
        # thrust back at medium acceleration
        ship_thrust['BM'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[1,2]*-1, ship_thrust_genes_trimf[1,1]*-1, ship_thrust_genes_trimf[1,0]*-1])
        # thrust back at low acceleration
        ship_thrust['BL'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[0,2]*-1, ship_thrust_genes_trimf[0,1]*-1, ship_thrust_genes_trimf[0,0]*-1])
        # thrust low value 
        ship_thrust['L'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[0,2]*-1/3, 0, ship_thrust_genes_trimf[0,2]*1/3])
        # thrust forward at low acceleration
        ship_thrust['FL'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[0,0], ship_thrust_genes_trimf[0,1], ship_thrust_genes_trimf[0,2]])
        # thrust forward at medium acceleration
        ship_thrust['FM'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[1,0], ship_thrust_genes_trimf[1,1], ship_thrust_genes_trimf[1,2]])
        # thrust forward at high acceleration
        ship_thrust['FH'] = fuzz.trimf(ship_thrust.universe, [ship_thrust_genes_trimf[2,0], ship_thrust_genes_trimf[2,1], ship_thrust_genes_trimf[2,2]])

        rule1_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['L'] & theta_diff['S'], ship_thrust['BH'])
        rule2_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['M'] & theta_diff['S'], ship_thrust['BM'])
        rule3_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['S'] & theta_diff['S'], ship_thrust['BM'])

        rule4_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['L'] & theta_diff['S'], ship_thrust['BM'])
        rule5_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['M'] & theta_diff['S'], ship_thrust['BM'])
        rule6_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['S'] & theta_diff['S'], ship_thrust['BL'])

        rule7_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['L'] & theta_diff['S'], ship_thrust['BM'])
        rule8_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['M'] & theta_diff['S'], ship_thrust['BL'])
        rule9_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['S'] & theta_diff['S'], ship_thrust['BL'])

        rule10_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['L'] & theta_diff['L'], ship_thrust['FH'])
        rule11_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['M'] & theta_diff['L'], ship_thrust['FM'])
        rule12_thrust = ctrl.Rule(asteroid_distance['S'] & asteroid_vel['S'] & theta_diff['L'], ship_thrust['FM'])

        rule13_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['L'] & theta_diff['L'], ship_thrust['FM'])
        rule14_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['M'] & theta_diff['L'], ship_thrust['FM'])
        rule15_thrust = ctrl.Rule(asteroid_distance['M'] & asteroid_vel['S'] & theta_diff['L'], ship_thrust['FL'])

        rule16_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['L'] & theta_diff['L'], ship_thrust['FM'])
        rule17_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['M'] & theta_diff['L'], ship_thrust['FL'])
        rule18_thrust = ctrl.Rule(asteroid_distance['L'] & asteroid_vel['S'] & theta_diff['L'], ship_thrust['FL'])

        rule19_thrust = ctrl.Rule(theta_diff['PM'], ship_thrust['BL'])
        rule20_thrust = ctrl.Rule(theta_diff['NM'], ship_thrust['BL'])

        self.thrust_movement = ctrl.ControlSystem()
        self.thrust_movement.addrule(rule1_thrust)
        self.thrust_movement.addrule(rule2_thrust)
        self.thrust_movement.addrule(rule3_thrust)
        self.thrust_movement.addrule(rule4_thrust)
        self.thrust_movement.addrule(rule5_thrust)
        self.thrust_movement.addrule(rule6_thrust)
        self.thrust_movement.addrule(rule7_thrust)
        self.thrust_movement.addrule(rule8_thrust)
        self.thrust_movement.addrule(rule9_thrust)

        self.thrust_movement.addrule(rule10_thrust)
        self.thrust_movement.addrule(rule11_thrust)
        self.thrust_movement.addrule(rule12_thrust)
        self.thrust_movement.addrule(rule13_thrust)
        self.thrust_movement.addrule(rule14_thrust)
        self.thrust_movement.addrule(rule15_thrust)
        self.thrust_movement.addrule(rule16_thrust)
        self.thrust_movement.addrule(rule17_thrust)
        self.thrust_movement.addrule(rule18_thrust)
        self.thrust_movement.addrule(rule19_thrust)
        self.thrust_movement.addrule(rule20_thrust)

    @staticmethod
    def scale_theta(val):
        new = (val * 2*math.pi) - math.pi
        return new

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?

        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity.
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.


        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)

            else:
                # closest_asteroid exists, and is thus initialized.
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object.
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.

        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!


        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)

        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        print("asteroid_vel: ", asteroid_vel)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)


        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta

        shooting.compute()

        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn'] * 3

        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        # And return your three outputs to the game simulation. Controller algorithm complete.
        # thrust = 0.0

        # thrust controller

        # finds angle between ship's heading and ship to asteroid
        x_diff = closest_asteroid["aster"]["position"][0] - ship_pos_x
        y_diff = closest_asteroid["aster"]["position"][1] - ship_pos_y
        ship_to_asteroid_theta = math.atan2(y_diff, x_diff)
        theta_diff = ship_to_asteroid_theta - ((math.pi/180)*ship_state["heading"])
        theta_diff = (theta_diff + math.pi) % (2 * math.pi) - math.pi

        asteroid_distance = closest_asteroid["dist"]
        movement = ctrl.ControlSystemSimulation(self.thrust_movement, flush_after_run=1)
        movement.input['asteroid_distance'] = asteroid_distance
        movement.input['asteroid_vel'] = asteroid_vel
        movement.input['theta_diff'] = theta_diff
        movement.compute()

        # gets magnitude of thrust
        thrust = movement.output['ship_thrust'] * 250

        drop_mine = False
        try:
            can_deploy = ship_state["can_deploy_mine"]
        except (KeyError, TypeError):
            can_deploy = False
        try:
            mines_remaining = ship_state["mines_remaining"]
        except (KeyError, TypeError):
            mines_remaining = 0

        # Cooldown: only drop 1 mine every 3 seconds (90 frames)
        frames_since_last_mine = self.eval_frames - self.last_mine_frame
        mine_cooldown_ready = frames_since_last_mine > 90

        # Fuzzy mine deployment
        rel_speed_val = 0.0
        try:
            to_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
            to_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
            if closest_asteroid["dist"] > 1:
                rel_speed_val = (closest_asteroid["aster"]["velocity"][0] * to_ship_x + closest_asteroid["aster"]["velocity"][1] * to_ship_y) / closest_asteroid["dist"]
            mine_sim = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
            mine_sim.input['mine_distance'] = closest_asteroid["dist"]
            mine_sim.input['rel_speed'] = rel_speed_val
            mine_sim.compute()
            mine_score = mine_sim.output.get('mine_decision', 0)
        except Exception:
            mine_score = 0
        if np.isnan(mine_score):
            mine_score = 0

        drop_window = closest_asteroid["dist"] < 200  # slightly larger window to allow drops
        approaching = rel_speed_val > 0
        if can_deploy and mines_remaining != 0 and mine_cooldown_ready and drop_window and (mine_score > 0.5 or (approaching and closest_asteroid["dist"] < 120)):
            drop_mine = True
            self.last_mine_frame = self.eval_frames

        # If we just dropped a mine (within last 1 second), turn and thrust away from nearest asteroid
        if frames_since_last_mine < 30:
            escape_heading = math.degrees(math.atan2(-y_diff, -x_diff)) % 360
            heading_diff = (escape_heading - ship_state["heading"] + 540) % 360 - 180
            turn_rate = max(min(heading_diff * 3, 180), -180)  # steer toward open space, clamp to ship limits
            thrust = 300

        self.eval_frames += 1

        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Custom Controller"
    
