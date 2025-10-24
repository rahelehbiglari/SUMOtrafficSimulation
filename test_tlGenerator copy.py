import random
import xml.etree.ElementTree as Et

def generate_individual(input_file, target_sum=90):
    # Parse the XML file
    tree = Et.parse(input_file)
    root = tree.getroot()
    
    phase_durations_individual = []
    
    # Find all traffic light logics
    traffic_lights = root.findall(".//tlLogic")
    for tl in traffic_lights:
        phases = tl.findall("phase")
        durations = []
        
        # Assign random durations for each phase
        for phase in phases:
            if "y" not in phase.get("state"):  # Non-yellow phases
                duration = random.randint(10, 84)
            else:  # Yellow phases
                duration = random.randint(3, 6)
            durations.append(duration)
        
        # Adjust durations to meet the target sum
        current_sum = sum(durations)
        while current_sum != target_sum:
            index = random.randint(0, len(durations) - 1)
            adjustment = random.choice([-1, 1])
            new_duration = durations[index] + adjustment
            if 10 <= new_duration <= 84 or (3 <= new_duration <= 6 and "y" in phases[index].get("state")):
                durations[index] = new_duration
                current_sum = sum(durations)
        
        # Add durations to the final individual
        phase_durations_individual.extend(durations)

        # Sanity check
        if len(phase_durations_individual) < 10:
            print("Error: Generate individuals mismatch ~~~~~~~~~~")

    return phase_durations_individual


# Input file
input_file = "Data/map.net.xml"

# Run the function
try:
    individual = generate_individual(input_file, target_sum=90)
    print("Generated Individual:", individual)
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Please ensure the file exists and the path is correct.")
