import random

import torch
from deap import base, creator, tools, algorithms
import xml.etree.ElementTree as Et
import queue

from Trainer import MLP
# Define the simple model structure
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(8096, 1)  # Ensure the input dimension matches the trained model

    def forward(self, x):
        return self.linear(x)

# Load the saved model
# model_path = "simple_model_HighLowInput.pth"
model_path = "simple_model_HighLowInputSameTL.pth"
model = LinearRegression()
model.load_state_dict(torch.load(model_path))  # Load trained weights
model.eval()  # Set the model to evaluation mode


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
            [10, 84],  # general duration range
            [3, 6],  # yellow light duration range
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

    def generate_individual(self,target_sum=90 ):
            # Initialize duration ranges
        min_yellow_duration = 3
        max_yellow_duration = 6
        min_duration = 10
        max_duration = 84
        """generates and returns an individual of random traffic light configuration based on a net file"""

        traffic_lights = self.root.findall(".//tlLogic")
        for tl in traffic_lights:        
            phase_durations = []
            phases = tl.findall("phase")
            
            for phase in phases:
                initial_duration = int(phase.get("duration", 0))  # Default to 0 if not set

                # Check if initial_duration falls in the "yellow" range
                if min_yellow_duration <= initial_duration <= max_yellow_duration:
                    duration = random.randint(min_yellow_duration, max_yellow_duration)
                else:
                    duration = random.randint(min_duration, max_duration)
                
                # Store and set the initial duration
                phase_durations.append(duration)
                phase.set("duration", str(duration))

            # Adjust durations to achieve the target sum for this tlLogic
            current_sum = sum(phase_durations)
            
            while current_sum != target_sum:
                # Randomly pick an index to adjust
                index = random.randint(0, len(phase_durations) - 1)
                adjustment = random.choice([-1, 1])  # Small adjustment to reach the target

                is_yellow_phase = min_yellow_duration <= phase_durations[index] <= max_yellow_duration

                # Apply adjustment within the allowed range
                if is_yellow_phase:
                    new_value = phase_durations[index] + adjustment
                    if min_yellow_duration <= new_value <= max_yellow_duration:
                        phase_durations[index] = new_value
                else:
                    new_value = phase_durations[index] + adjustment
                    if min_duration <= new_value <= max_duration:
                        phase_durations[index] = new_value

                # Recalculate the sum 
                current_sum = sum(phase_durations)

            # Set the final adjusted durations
            # for idx, phase in enumerate(phases):
            #     phase.set("duration", str(phase_durations[idx]))
        return phase_durations

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

    # change this function!!
    def objective_function(self, individual):
        """
        function the genetic algorithm will minimise
        """
        situation = torch.tensor(individual + self.car_list, dtype=torch.float32)
        if self.density <= 275: # which model to use
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
        # Initialize duration ranges
        target_sum = 90
        min_yellow_duration = 3
        max_yellow_duration = 6
        min_duration = 10
        max_duration = 84
        configs = []
        traffic_lights = self.root.findall(".//tlLogic")
        for tl in traffic_lights:        
            configs = []
            phases = tl.findall("phase")
            
            for phase in phases:
                initial_duration = int(phase.get("duration", 0))  # Default to 0 if not set

                # Check if initial_duration falls in the "yellow" range
                if min_yellow_duration <= initial_duration <= max_yellow_duration:
                    duration = random.randint(min_yellow_duration, max_yellow_duration)
                else:
                    duration = random.randint(min_duration, max_duration)
                
                # Store and set the initial duration
                configs.append(duration)
                phase.set("duration", str(duration))

            # Adjust durations to achieve the target sum for this tlLogic
            current_sum = sum(configs)
            
            while current_sum != target_sum:
                # Randomly pick an index to adjust
                index = random.randint(0, len(configs) - 1)
                adjustment = random.choice([-1, 1])  # Small adjustment to reach the target

                is_yellow_phase = min_yellow_duration <= phase_durations[index] <= max_yellow_duration

                # Apply adjustment within the allowed range
                if is_yellow_phase:
                    new_value = phase_durations[index] + adjustment
                    if min_yellow_duration <= new_value <= max_yellow_duration:
                        phase_durations[index] = new_value
                else:
                    new_value = phase_durations[index] + adjustment
                    if min_duration <= new_value <= max_duration:
                        phase_durations[index] = new_value

                # Recalculate the sum 
                current_sum = sum(phase_durations)

            # Set the final adjusted durations
            # for idx, phase in enumerate(phases):
            #     phase.set("duration", str(phase_durations[idx]))
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

