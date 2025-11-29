import EasyGA
import random
import numpy as np
from kesslergame import Scenario, GraphicsType, KesslerGame, TrainerEnvironment
from custom_controller_ga import CustomController

def fitness(chromosome):
    chromosome = [gene.value for gene in chromosome]

    my_test_scenario = Scenario(name='Test Scenario',
        num_asteroids=10,
        ship_states=[
            {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
        ],
        map_size=(1000, 800),
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False)
    
    game_settings = {'perf_tracker': True,
        'graphics_type': GraphicsType.Tkinter,
        'realtime_multiplier': 1,
        'graphics_obj': None,
        'frequency': 30}

    game = TrainerEnvironment(settings=game_settings)

    controller = CustomController(chromosome)
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[controller])

    result = score.teams[0].asteroids_hit + score.teams[0].accuracy - score.teams[0].deaths

    return result

def gene_generation():
  return random.uniform(0, 1)

ga = EasyGA.GA()
ga.chromosome_length = X # need to add
ga.population_size = 20
ga.target_fitness_type = 'max'
ga.generation_goal = 20
ga.fitness_function_impl = fitness
ga.gene_impl = lambda: gene_generation()

ga.database_name = ""

ga.evolve()
ga.print_best_chromosome()
