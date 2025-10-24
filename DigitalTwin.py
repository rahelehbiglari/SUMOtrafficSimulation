import random

import torch
from deap import base, creator, tools, algorithms
import xml.etree.ElementTree as Et
import queue

from Trainer import MLP


class DigitalTwin:
    def __init__(self, net_file):
        """digital twin of the traffic system, contains the genetic optimiser as well as the MAPE control loop"""
        self.net = net_file
        self.tree = Et.parse(net_file)
        self.root = self.tree.getroot()
        self.density = None
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.car_list = []
        self.model_light = MLP(num_outputs=1, num_hidden=50, lr=0.0003)
        self.model_dense = MLP(num_outputs=1, num_hidden=50, lr=0.0003)
        self.traffic_light_range = [
            [10, 60],  # general duration range
            [2, 10],  # yellow light duration range
        ]

        self.model_light.load_state_dict(torch.load('mlpLowDense.params'), strict=False)
        self.model_light.eval()

        self.model_dense.load_state_dict(torch.load('mlpDense.params'), strict=False)
        self.model_dense.eval()

        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.objective_function)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)

    def generate_individual(self):
        """generates and returns an individual of random traffic light configuration based on a net file"""

        traffic_light_individual = []
        traffic_lights = self.root.findall(".//tlLogic")
        for tl in traffic_lights:
            for phase in tl.findall("phase"):
                if "y" in phase.get("state"):
                    traffic_light_individual.append(random.randint(self.traffic_light_range[1][0], self.traffic_light_range[1][1]))
                else:
                    traffic_light_individual.append(random.randint(self.traffic_light_range[0][0], self.traffic_light_range[0][1]))
        return traffic_light_individual

    def get_vehicles_per_road(self, state):
        tree_state = Et.parse(state)
        root_state = tree_state.getroot()

        vehicle_per_road = {}

        for edge in self.root.findall(".//edge"):
            for lane in edge.findall("lane"):
                if lane.get("allow") is not None:
                    if "passenger" in lane.attrib["allow"]:
                        lane_id = lane.attrib["id"]
                        vehicle_per_road[lane_id] = 0
                elif lane.attrib["disallow"] is not None:
                    if "passenger" not in lane.attrib["disallow"]:
                        lane_id = lane.attrib["id"]
                        vehicle_per_road[lane_id] = 0

        for lane in root_state.findall(".//lane"):
            lane_id = lane.attrib["id"]
            if lane.find("vehicles") is not None:
                vehicles = lane.find("vehicles").attrib["value"]
                vehicle_list = vehicles.split(" ") if " " in vehicles else [vehicles]
                if lane_id in vehicle_per_road.keys():
                    vehicle_per_road[lane_id] = len(vehicle_list)
                else:
                    print(lane_id)
            else:
                vehicle_per_road[lane_id] = len(lane.find("link"))

        return vehicle_per_road

    def objective_function(self, individual):
        """
        function the genetic algorithm will minimise
        """
        situation = torch.tensor(individual + self.car_list, dtype=torch.float32)
        if self.density <= 275:
            curr_model = self.model_light
        else:
            curr_model = self.model_dense
        curr_model.eval()
        with torch.no_grad():
            prediction = curr_model(situation)
        return (prediction.item(),)

    def mutate_individual(self, individual):
        """
        function to mutate an individual during the optimisation process
        """
        configs = []
        for tl in self.root.findall(".//tlLogic"):
            for phase in tl.findall("phase"):
                configs.append(phase)
        # mutate about a third of traffic light timings
        for i in range(0, len(individual)//3):
            index_to_mutate = random.randint(0, len(individual) - 1)
            if "y" in configs[index_to_mutate].get("state"):
                individual[index_to_mutate] = random.randint(self.traffic_light_range[1][0], self.traffic_light_range[1][1])
            else:
                individual[index_to_mutate] = random.randint(self.traffic_light_range[0][0], self.traffic_light_range[0][1])

        return (individual,)

    def optimize(self, state, q):
        """
        Function to call the digital twin. Will select a model, start the optimiser and return a new configuration set
        """
        self.car_list = list(self.get_vehicles_per_road(state).values())
        self.density = sum(self.car_list)
        population = self.toolbox.population(15)
        algorithms.eaSimple(population, self.toolbox, cxpb=0.7, mutpb=0.1, ngen=50, stats=None, verbose=True)

        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]
        print("Best individual:", best_individual)
        print("Best fitness value:", best_fitness)

        q.put(best_individual)

