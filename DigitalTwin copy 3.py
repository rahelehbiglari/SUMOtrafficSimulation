# wworking on 90s constraint
import random
import torch
from deap import base, creator, tools, algorithms
import xml.etree.ElementTree as Et
import queue
from trainer_ver2 import DeepNeuralNetwork  # Import new models

class DigitalTwin:
    def __init__(self, net_file):
        """Digital twin of the traffic system, uses Genetic Algorithm and Neural Network-based optimization"""
        self.net = net_file
        self.tree = Et.parse(net_file)
        self.root = self.tree.getroot()
        self.density = None
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.car_list = []
        
        # Load models
        # self.model_light = LinearRegression().to(torch.device("cpu"))
        self.model_dense = DeepNeuralNetwork().to(torch.device("cpu"))

        # self.model_light.load_state_dict(torch.load("simple_model_HighLowInputDiffTL.pth", map_location=torch.device("cpu")))
        self.model_dense.load_state_dict(torch.load("deep_model_HighLowInputDiffTL.pth", map_location=torch.device("cpu"), weights_only=True))

        # self.model_light.eval()
        self.model_dense.eval()

        self.traffic_light_range = [[10, 84], [3, 6]]  # [general, yellow duration]

        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.objective_function)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_individual)

    def generate_individual(self):
        """Generates a valid traffic light configuration where:
        - Total duration per intersection = 90s
        - Yellow phase duration: 3-6s
        - Other phases: 10-84s
        """
        traffic_light_individual = []
        traffic_lights = self.root.findall(".//tlLogic")

        for tl in traffic_lights:
            phases = tl.findall("phase")
            phase_durations = [0] * len(phases)  # Initialize durations
            remaining_time = 90  # Start with 90s total

            # Step 1: Assign Yellow Phases First
            yellow_indices = [i for i, phase in enumerate(phases) if "y" in phase.get("state")]
            for i in yellow_indices:
                phase_durations[i] = random.randint(3, 6)
                remaining_time -= phase_durations[i]

            # Step 2: Assign Other Phases
            non_yellow_indices = [i for i in range(len(phases)) if i not in yellow_indices]
            if not non_yellow_indices:
                raise ValueError("⚠️ Error: No non-yellow phases found!")

            # Ensure total sum is exactly 90s
            min_needed = len(non_yellow_indices) * 10  # Each must be at least 10
            max_possible = len(non_yellow_indices) * 84  # Each at most 84
            remaining_time = max(min_needed, min(remaining_time, max_possible))

            # Step 3: Distribute remaining time across non-yellow phases
            splits = sorted(random.sample(range(min_needed, remaining_time), len(non_yellow_indices) - 1))
            splits = [0] + splits + [remaining_time]

            for i in range(len(non_yellow_indices)):
                phase_durations[non_yellow_indices[i]] = splits[i + 1] - splits[i]

            # Step 4: Final Adjustment to Ensure Sum = 90s
            while sum(phase_durations) != 90:
                diff = 90 - sum(phase_durations)
                adjust_index = random.choice(non_yellow_indices)
                new_value = phase_durations[adjust_index] + diff

                if 10 <= new_value <= 84:
                    phase_durations[adjust_index] = new_value
                else:
                    # If the new value is out of bounds, adjust the closest valid value
                    phase_durations[adjust_index] = max(10, min(84, new_value))

            # Add the finalized phase durations to the individual
            traffic_light_individual.extend(phase_durations)

            # Verify final sum (should always be 90)
            assert sum(phase_durations) == 90, f" Error: Incorrect sum {sum(phase_durations)}, expected 90."

        return traffic_light_individual


        # """Generates an individual traffic light configuration with constraints:
        # - Total duration per intersection = 90s
        # - Yellow phase duration: 3-6s
        # - Other phases: 10-84s
        # """
        # traffic_light_individual = []
        # traffic_lights = self.root.findall(".//tlLogic")

        # for tl in traffic_lights:
        #     phases = tl.findall("phase")
        #     phase_durations = []

        #     # Generate initial random durations
        #     for phase in phases:
        #         if "y" in phase.get("state"):  # If yellow phase
        #             duration = random.randint(3, 6)
        #         else:
        #             duration = random.randint(10, 84)
        #         phase_durations.append(duration)

        #     # Adjust to meet the total target sum of 90s
        #     current_sum = sum(phase_durations)
        #     while current_sum != 90:
        #         index = random.randint(0, len(phase_durations) - 1)
        #         adjustment = random.choice([-1, 1])

        #         # Determine if it's a yellow phase or normal phase
        #         is_yellow = "y" in phases[index].get("state")

        #         # Ensure adjustments stay within limits
        #         if is_yellow:
        #             new_duration = phase_durations[index] + adjustment
        #             if 3 <= new_duration <= 6:
        #                 phase_durations[index] = new_duration
        #         else:
        #             new_duration = phase_durations[index] + adjustment
        #             if 10 <= new_duration <= 84:
        #                 phase_durations[index] = new_duration

        #         # Recalculate total duration
        #         current_sum = sum(phase_durations)

        #     # Add finalized phase durations to individual
        #     traffic_light_individual.extend(phase_durations)

        # return traffic_light_individual

        # """Generates an individual traffic light configuration"""
        # traffic_light_individual = []
        # for tl in self.root.findall(".//tlLogic"):
        #     for phase in tl.findall("phase"):
        #         if "y" in phase.get("state"):
        #             traffic_light_individual.append(random.randint(self.traffic_light_range[1][0], self.traffic_light_range[1][1]))
        #         else:
        #             traffic_light_individual.append(random.randint(self.traffic_light_range[0][0], self.traffic_light_range[0][1]))
        # return traffic_light_individual
        

    def get_vehicles_per_road(self, state):
        tree_state = Et.parse(state)
        root_state = tree_state.getroot()

        vehicle_per_road = {}
        for edge in self.root.findall(".//edge"):
            for lane in edge.findall("lane"):
                if lane.get("allow") is not None and "passenger" in lane.attrib["allow"]:
                    vehicle_per_road[lane.attrib["id"]] = 0

        for lane in root_state.findall(".//lane"):
            lane_id = lane.attrib["id"]
            if lane.find("vehicles") is not None:
                vehicles = lane.find("vehicles").attrib["value"]
                vehicle_list = vehicles.split(" ") if " " in vehicles else [vehicles]
                vehicle_per_road[lane_id] = len(vehicle_list)

        return vehicle_per_road

    def objective_function(self, individual):
        """Evaluates fitness using trained neural network models"""
        situation = torch.tensor(individual + self.car_list, dtype=torch.float32)

        # Ensure correct input size (8096)
        if situation.shape[0] < 8096:
            padding = torch.zeros(8096 - situation.shape[0])  # Zero padding for missing values
            situation = torch.cat((situation, padding))

        situation = situation.unsqueeze(0)  # Add batch dimension

        # curr_model = self.model_light if self.density <= 275 else self.model_dense
        curr_model = self.model_dense if self.density <= 275 else self.model_dense
        curr_model.eval()

        with torch.no_grad():
            prediction = curr_model(situation)

        return (prediction.item(),)

    def mutate_individual(self, individual):
        """Mutates an individual traffic light configuration while maintaining constraints:
        - Yellow phase duration: 3-6s
        - Other phases: 10-84s
        - Total duration per intersection = 90s
        """
        configs = []
        for tl in self.root.findall(".//tlLogic"):
            for phase in tl.findall("phase"):
                configs.append(phase)

        # Mutate about 1/3 of the durations
        for _ in range(len(individual) // 3):
            index_to_mutate = random.randint(0, len(individual) - 1)

            # Determine if it's a yellow phase or normal phase
            is_yellow = "y" in configs[index_to_mutate].get("state")

            # Apply mutation within allowed range
            if is_yellow:
                individual[index_to_mutate] = random.randint(3, 6)
            else:
                individual[index_to_mutate] = random.randint(10, 84)

        # Ensure the total remains 90s
        current_sum = sum(individual)
        while current_sum != 90:
            index = random.randint(0, len(individual) - 1)
            adjustment = random.choice([-1, 1])

            is_yellow = "y" in configs[index].get("state")

            # Apply adjustment while keeping constraints
            if is_yellow:
                new_value = individual[index] + adjustment
                if 3 <= new_value <= 6:
                    individual[index] = new_value
            else:
                new_value = individual[index] + adjustment
                if 10 <= new_value <= 84:
                    individual[index] = new_value

            # Recalculate the sum
            current_sum = sum(individual)

        return (individual,)

        # """Mutates an individual during the optimization process"""
        # configs = []
        # for tl in self.root.findall(".//tlLogic"):
        #     for phase in tl.findall("phase"):
        #         configs.append(phase)

        # for _ in range(len(individual) // 3):
        #     index_to_mutate = random.randint(0, len(individual) - 1)
        #     if "y" in configs[index_to_mutate].get("state"):
        #         individual[index_to_mutate] = random.randint(self.traffic_light_range[1][0], self.traffic_light_range[1][1])
        #     else:
        #         individual[index_to_mutate] = random.randint(self.traffic_light_range[0][0], self.traffic_light_range[0][1])

        # return (individual,)

    # def optimize(self, state, q):
    #     """Runs genetic optimization to find the best traffic light settings"""
    #     self.car_list = list(self.get_vehicles_per_road(state).values())
    #     self.density = sum(self.car_list)

    #     population = self.toolbox.population(15)
    #     algorithms.eaSimple(population, self.toolbox, cxpb=0.7, mutpb=0.1, ngen=50, stats=None, verbose=True)

    #     best_individual = tools.selBest(population, k=1)[0]
    #     print("Best individual:", best_individual)
    #     q.put(best_individual)
    def optimize(self, state):
        """Runs genetic optimization to find the best traffic light settings"""
        self.car_list = list(self.get_vehicles_per_road(state).values())
        self.density = sum(self.car_list)

        population = self.toolbox.population(15)
        algorithms.eaSimple(population, self.toolbox, cxpb=0.7, mutpb=0.1, ngen=50, stats=None, verbose=True)

        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]

        print("\n✅ Optimization Completed!")
        print(f"🚦 Best Traffic Light Configuration: {best_individual}")
        print(f"🏆 Best Fitness Value: {best_fitness}")

        return best_individual, best_fitness
#  Initialize the digital twin with the SUMO network file
net_file = "Data/map.net.xml"
digital_twin = DigitalTwin(net_file)

# Run optimization with the SUMO traffic state file
state_file = "Data/training_data/states/states_0/_100.00.xml"
best_config, best_fitness = digital_twin.optimize(state_file)

# Print the best result
print("\n✅ Final Optimized Traffic Light Settings:")
print(f"🚦 Best Configuration: {best_config}")
print(f"🏆 Fitness Value (Lower is Better): {best_fitness}")
