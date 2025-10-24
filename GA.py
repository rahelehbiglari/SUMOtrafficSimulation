import random
import xml.etree.ElementTree as Et
import torch
import torch.nn as nn
import numpy as np
from trainer_Claudio import LinearRegression

class TrafficLightOptimizer:
    def __init__(self, net_file_path):
        self.density = None
        self.car_list = []
        self.modelHighLow = LinearRegression()
        # self.modelHigh = LinearRegression()
        model_path = "deep_model_HighLowInputDiffTL.pth"
        self.modelHighLow.load_state_dict(torch.load(model_path))
        self.modelHighLow.eval()
        self.tree = Et.parse(net_file_path)
        self.root = self.tree.getroot()
        self.traffic_light_range = [(10, 84), (3, 6)]  # [non-yellow range, yellow range]

    def generate_individual(self, target_sum=90):
        phase_durations_individual = []
        traffic_lights = self.root.findall(".//tlLogic")
        for tl in traffic_lights:
            phases = tl.findall("phase")
            durations = []
            for phase in phases:
                duration = random.randint(10, 84) if "y" not in phase.get("state") else random.randint(3, 6)
                durations.append(duration)
            # Adjust durations to meet the target_sum
            current_sum = sum(durations)
            while current_sum != target_sum:
                index = random.randint(0, len(durations) - 1)
                adjustment = random.choice([-1, 1])
                new_duration = durations[index] + adjustment
                if 10 <= new_duration <= 84 or (3 <= new_duration <= 6 and "y" in phases[index].get("state")):
                    durations[index] = new_duration
                    current_sum = sum(durations)
            phase_durations_individual.extend(durations)
            if len(phase_durations_individual) < 10:
                print("Error: Generate individuals mismatch ~~~~~~~~~~")
        return phase_durations_individual

    def objective_function(self, individual):
        """
        function the genetic algorithm will minimise
        """
        situation = torch.tensor(individual + self.car_list, dtype=torch.float32)
        if len(situation) < 8096:
            print("Error: mismatch ~~~~~~~~~~")
        # if self.density <= 275:
        #     curr_model = self.model_light
        # else:
        #     curr_model = self.model_dense
        # curr_model.eval()
        curr_model = self.modelHighLow
        with torch.no_grad():
            prediction = curr_model(situation)
        return (prediction.item(),)
    
    def crossover(self, parent1, parent2):
        """Performs single-point crossover between two parents."""
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def select(self, population, fitness_scores, num_parents):
        """Selects the best individuals based on fitness scores."""
        # sorted_indices = np.argsort(fitness_scores.tolist())
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])

        selected = [population[i] for i in sorted_indices[:num_parents]]
        return selected
    
    # def mutate_individual(self, individual):
    #         """
    #         function to mutate an individual during the optimisation process
    #         """
    #                     # Initialize duration ranges
    #         min_yellow_duration = 3
    #         max_yellow_duration = 6
    #         min_duration = 10
    #         max_duration = 84
    #         target_sum = 90
    #         """generates and returns an individual of random traffic light configuration based on a net file"""

    #         traffic_lights = self.root.findall(".//tlLogic")
    #         for tl in traffic_lights:        
    #             phase_durations_individual = []
    #             configs = []
    #             phases = tl.findall("phase")
                
    #             for phase in phases:
    #                 configs.append(phase)
    #                 # mutate about a third of traffic light timings
    #                 for i in range(0, len(individual)//3):
    #                     index_to_mutate = random.randint(0, len(individual) - 1)
    #                     # Ensure `index_to_mutate` is valid for both `configs` and `individual`
    #                     if index_to_mutate >= len(configs):
    #                         continue  # Skip invalid indices
    #                     current_duration = int(configs[index_to_mutate].get("duration", 0))
    #                     # Check if initial_duration falls in the "yellow" range
    #                     if min_yellow_duration <= current_duration <= max_yellow_duration:
    #                         new_duration  = random.randint(min_yellow_duration, max_yellow_duration)
    #                     else:
    #                         new_duration  = random.randint(min_duration, max_duration)
    #                     # Apply the mutation
    #                     individual[index_to_mutate] = new_duration
                    
    #                 # # Store and set the initial duration
    #                 # phase_durations_individual.append(duration)
    #                 # phase.set("duration", str(duration))

    #             # Adjust durations to achieve the target sum for this tlLogic
    #             current_sum = sum(individual)
                
    #             while current_sum != target_sum:
    #                 # Randomly pick an index to adjust
    #                 index = random.randint(0, len(individual) - 1)
    #                 adjustment = random.choice([-1, 1])  # Small adjustment to reach the target

    #                 is_yellow_phase = min_yellow_duration <= individual[index] <= max_yellow_duration

    #                 # Apply adjustment within the allowed range
    #                 if is_yellow_phase:
    #                     new_value = individual[index] + adjustment
    #                     if min_yellow_duration <= new_value <= max_yellow_duration:
    #                         individual[index] = new_value
    #                 else:
    #                     new_value = individual[index] + adjustment
    #                     if min_duration <= new_value <= max_duration:
    #                         individual[index] = new_value

    #                 # Recalculate the sum 
    #                 current_sum = sum(individual)

    #         return (individual,)
    def mutate_individual(self, individual):
        """
        Mutates an individual by altering some of the traffic light timings
        while ensuring constraints like the target sum are respected.
        """
        # Initialize duration ranges
        min_yellow_duration = 3
        max_yellow_duration = 6
        min_duration = 10
        max_duration = 84
        target_sum = 90

        # Determine the number of mutations to perform
        num_mutations = max(1, len(individual) // 3)  # Mutate about a third of the durations

        # Mutate selected genes
        for _ in range(num_mutations):
            index_to_mutate = random.randint(0, len(individual) - 1)

            # Determine if the phase being mutated is a yellow light phase
            if min_yellow_duration <= individual[index_to_mutate] <= max_yellow_duration:
                # Mutate within yellow light range
                new_value = random.randint(min_yellow_duration, max_yellow_duration)
            else:
                # Mutate within non-yellow range
                new_value = random.randint(min_duration, max_duration)

            individual[index_to_mutate] = new_value

        # Adjust to ensure the total duration matches the target sum
        current_sum = sum(individual)
        while current_sum != target_sum:
            # Select a random index to adjust
            index = random.randint(0, len(individual) - 1)
            adjustment = random.choice([-1, 1])  # Small adjustment to approach target sum
            new_value = individual[index] + adjustment

            # Ensure the adjustment respects the duration constraints
            if (min_yellow_duration <= new_value <= max_yellow_duration or
                    min_duration <= new_value <= max_duration):
                individual[index] = new_value
                current_sum = sum(individual)  # Recalculate the total

        return individual


    def optimize(self, state, num_generations=2, population_size=5, mutation_rate=0.1):
        """
        Runs the genetic algorithm to optimize traffic light configurations.
        """
        # Initialize
        self.car_list = list(self.get_vehicles_per_road(state).values())
        self.density = sum(self.car_list)

        # Generate initial population
        population = [self.generate_individual() for _ in range(population_size)]

        for generation in range(num_generations):
            # Evaluate fitness
            fitness_scores = [self.objective_function(individual) for individual in population]

            # Print the best fitness of this generation
            best_fitness = min(fitness_scores)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

            # Select parents
            parents = self.select(population, fitness_scores, num_parents=population_size // 2)

            # Crossover to create children
            next_population = []
            while len(next_population) < population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.extend([child1, child2])

            # Mutate some individuals
            for i in range(len(next_population)):
                if random.random() < mutation_rate:
                    next_population[i] = self.mutate_individual(next_population[i])

            # Update population
            population = next_population[:population_size]

        # Return the best individual
        fitness_scores = [self.objective_function(individual) for individual in population]
        best_index = np.argmin(fitness_scores)
        best_individual = population[best_index]
        print("Best Individual:", best_individual)
        print("Best Fitness Value:", fitness_scores[best_index])
        return best_individual
    
    def get_vehicles_per_road(self, state):
        """Parses the traffic state and returns the number of vehicles per road."""
        tree_state = Et.parse(state)
        root_state = tree_state.getroot()
        vehicle_per_road = {}

        for edge in self.root.findall(".//edge"):
            for lane in edge.findall("lane"):
                if lane.get("allow") is not None:
                    if "passenger" in lane.attrib["allow"]:
                        vehicle_per_road[lane.attrib["id"]] = 0
                elif lane.attrib["disallow"] is not None:
                    if "passenger" not in lane.attrib["disallow"]:
                        vehicle_per_road[lane.attrib["id"]] = 0

        for lane in root_state.findall(".//lane"):
            lane_id = lane.attrib["id"]
            vehicles = lane.find("vehicles")
            if vehicles is not None:
                vehicle_list = vehicles.attrib["value"].split(" ") if " " in vehicles.attrib["value"] else [vehicles.attrib["value"]]
                vehicle_per_road[lane_id] = len(vehicle_list)
            else:
                vehicle_per_road[lane_id] = 0

        return vehicle_per_road
    
# # Create the optimizer instance with the path to the net file
# optimizer = TrafficLightOptimizer("Data/map.net.xml")

# # Generate an individual
# individual = optimizer.generate_individual(target_sum=90)
# print("Generated Individual:", individual)

# # Mutate the individual
# mutated_individual = optimizer.mutate_individual(individual)
# print("Mutated Individual:", mutated_individual)

# Initialize the digital twin
net_file = "Data/map.net.xml"
digital_twin = TrafficLightOptimizer(net_file)

# Run optimization
state_file = "Data/training_data/states/states_0/_100.00.xml"
# best_config = digital_twin.optimize(state_file, num_generations=50, population_size=15)
best_config = digital_twin.optimize(state_file, num_generations=2, population_size=5)

