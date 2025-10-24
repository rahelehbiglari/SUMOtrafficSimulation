# works but the problem is with 90 seconds constraint
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
        """Generates an individual traffic light configuration"""
        traffic_light_individual = []
        for tl in self.root.findall(".//tlLogic"):
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
        """Mutates an individual during the optimization process"""
        configs = []
        for tl in self.root.findall(".//tlLogic"):
            for phase in tl.findall("phase"):
                configs.append(phase)

        for _ in range(len(individual) // 3):
            index_to_mutate = random.randint(0, len(individual) - 1)
            if "y" in configs[index_to_mutate].get("state"):
                individual[index_to_mutate] = random.randint(self.traffic_light_range[1][0], self.traffic_light_range[1][1])
            else:
                individual[index_to_mutate] = random.randint(self.traffic_light_range[0][0], self.traffic_light_range[0][1])

        return (individual,)

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
