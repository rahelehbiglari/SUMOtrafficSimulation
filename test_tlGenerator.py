import random
import time


class TrafficLightOptimizer:
    def __init__(self):
        self.min_yellow_duration = 3
        self.max_yellow_duration = 6
        self.min_duration = 10
        self.max_duration = 84
        self.target_sum = 90

    def mutate_individual(self, individual):
        """
        Mutates an individual by altering some of the traffic light timings
        while ensuring constraints like the target sum are respected.
        """
        # Duration ranges
        min_yellow_duration = self.min_yellow_duration
        max_yellow_duration = self.max_yellow_duration
        min_duration = self.min_duration
        max_duration = self.max_duration
        target_sum = self.target_sum

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


def main():
    # Create an instance of the TrafficLightOptimizer
    optimizer = TrafficLightOptimizer()

    # Generate a random individual
    num_phases = 10  # Number of traffic light phases
    individual = [10,80
    ]
    # Ensure the initial individual meets the target sum constraint
    adjustment = optimizer.target_sum - sum(individual)
    if adjustment != 0:
        individual[0] += adjustment

    print("Original Individual:", individual)
    print("Original Sum:", sum(individual))

    # Mutate the individual and measure the time taken
    start_time = time.time()
    mutated_individual = optimizer.mutate_individual(individual)
    end_time = time.time()

    print("Mutated Individual:", mutated_individual)
    print("Mutated Sum:", sum(mutated_individual))
    print(f"Time Taken: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
