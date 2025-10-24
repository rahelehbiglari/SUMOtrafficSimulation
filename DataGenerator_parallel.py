import csv
import subprocess
import random
import sys
import xml.etree.ElementTree as Et
from pathlib import Path

import time
import os
import multiprocessing as mp

import traci
from sumolib import checkBinary


def generate_random_traffic(net, output_routes, output_trips, num_vehicles, density):
    """Uses randomTrips.py to generate a set of random trips for a given net file"""
    command = [
        "python",
        "tools/randomTrips.py",
        "-n", net,
        "-r", output_routes,
        "-o", output_trips,
        "-e", str(num_vehicles),
        "--random",
        "-l", "-L",
        "--binomial", "5",
        "--insertion-density", str(density)
    ]
    try:
        subprocess.run(command, check=True)
        print("Random traffic generated successfully.")
    except subprocess.CalledProcessError as e:
        print("Error generating random traffic:", e)

# it could be better in case of performance, still works
def generate_random_traffic_lights(input_net_file, output_net_file, target_sum=90):
    """Generate random traffic light configuration for each tlLogic based on a net file and save it to output_net_file."""
    tree = Et.parse(input_net_file)
    root = tree.getroot()

    # Initialize duration ranges
    min_yellow_duration = 3
    max_yellow_duration = 6
    min_duration = 10
    max_duration = 84


    traffic_lights = root.findall(".//tlLogic")
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
        for idx, phase in enumerate(phases):
            phase.set("duration", str(phase_durations[idx]))

    tree.write(output_net_file)

#replaced with the new random traffic light generator
def old_generate_random_traffic_lights(input_net_file, output_net_file):
    """generate random traffic light configuration based on a net file and stores a new net file in the
    output_net_file location"""
    tree = Et.parse(input_net_file)
    root = tree.getroot()

    min_yellow_duration = 2
    max_yellow_duration = 10
    min_duration = 10
    max_duration = 60

    traffic_lights = root.findall(".//tlLogic")
    for tl in traffic_lights:
        for phase in tl.findall("phase"):
            if "y" in phase.get("state"):
                phase.set("duration", str(random.randint(min_yellow_duration, max_yellow_duration)))
            else:
                phase.set("duration", str(random.randint(min_duration, max_duration)))
    tree.write(output_net_file)


def get_curr_traffic_light_config(input_net_file):
    """Obtain the curren traffic light configuration from a net file"""
    tree = Et.parse(input_net_file)
    root = tree.getroot()

    traffic_light_config = []
    for tl in root.findall(".//tlLogic"):
        for phase in tl.findall('phase'):
            traffic_light_config.append(int(phase.get("duration")))
    return traffic_light_config


def generate_sumo_cfg(input_net_file, input_rou_file, output_sumo_file):
    """
    Use a net file and a rou file to generate a sumocfg file
    """
    with open(output_sumo_file, 'w') as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on TIME by Eclipse SUMO sumo Version 1.18.0
-->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{input_net_file}"/>
        <route-files value="{input_rou_file}"/>
    </input>

</configuration>
    """)


def run(end_time):
    """execute the TraCI control loop"""
    step = 0

    while traci.simulation.getTime() < end_time:  # traci.simulation.getMinExpectedNumber() > 0:
        # by default a simulation step is 1 second.
        traci.simulationStep(step)
        step += 1
    # traci.simulation.saveState('data/states/statesavedState.xml')  # define when to save the state
    traci.close()
    sys.stdout.flush()


def get_vehicles_per_road(state, net):
    """
    Returns a list containing the active vehicles on every road in the system (that allows cars)
    """
    tree_state = Et.parse(state)
    tree_net = Et.parse(net)
    root_state = tree_state.getroot()
    root_net = tree_net.getroot()

    vehicle_per_road = {}

    for edge in root_net.findall(".//edge"):
        for lane in edge.findall("lane"):
            if lane.get("allow") is not None:
                if "passenger" in lane.attrib["allow"]:
                    lane_id = lane.attrib["id"]
                    edge_id = edge.attrib["id"]
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


def run_long_simulation(duration, set_number):
    """run a long simulation to generate set of state files"""
    state_folder =  f"data/training_data/states/states_{set_number}/"
    if not os.path.exists(state_folder):
        os.mkdir(state_folder)
    traci.start([checkBinary('sumo'), "-c", "data/training_data/set" + set_number + "_data.sumocfg", "--start",
                 "--quit-on-end", "--save-state.period", "100", "--save-state.suffix", ".xml",
                 "--save-state.prefix",
                 state_folder])
    time = traci.simulation.getTime()
    # run the simulation until 500 time steps before the last vehicle spawn
    run(time + duration)
    return state_folder


def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


    
def run_short_simulation_from_state(duration, set_number, state, statistics_file):
    """Run a simulation for a shorter period to generate a result based on a given state"""    
    statistics_folder = f"data/training_data/statistics/statistics_{set_number}/"
    if not os.path.exists(statistics_folder):
        os.mkdir(statistics_folder)
    traci.start([checkBinary('sumo'), "-c", "data/training_data/set" + set_number + "_data.sumocfg",
                 "--tripinfo-output", "data/sumo_output/tripinfo.xml",
                 "--tripinfo-output.write-unfinished", "True",
                 "--statistic-output", statistics_file,
                 "--start",
                 "--quit-on-end", "--load-state", state])
    
    time = traci.simulation.getTime()
    run(time + duration)
    


def get_waiting_time_from_statistics(statistics):
    tree_stat = Et.parse(statistics)
    root_stat = tree_stat.getroot()

    stats = root_stat.find("vehicleTripStatistics")
    waiting_time = [float(stats.attrib["waitingTime"])]
    return waiting_time
def copy_file(read_file, write_file):
    """generate random traffic light configuration based on a net file and stores a new net file in the
    output_net_file location"""
    tree = Et.parse(read_file)
    root = tree.getroot()
    tree.write(write_file)

# Function to parse the junction XML and extract junctions with their incoming lanes
def extract_junction_lanes(junction_file):
    # Parse the junction XML file
    junction_tree = Et.parse(junction_file)
    junction_root = junction_tree.getroot()

    # Create a list to store the junction data with incoming lanes
    junction_data = []

    # Extract junction data with incoming lanes
    for junction in junction_root.findall('junction'):
        junction_id = junction.get('id')
        inc_lanes = junction.get('incLanes').split()
        # Append each junction and its incoming lanes
        junction_data.append({
            "junction_id": junction_id,
            "inc_lanes": inc_lanes
        })

    return junction_data

# Function to parse the lane XML and extract lanes with vehicle counts
def extract_lanes_with_vehicles(lane_file):
    # Parse the lane XML file
    lane_tree = Et.parse(lane_file)
    lane_root = lane_tree.getroot()

    # Create a dictionary to map lane IDs to their vehicle counts
    lane_vehicle_count = {}

    # Extract vehicle data from lane tags
    for lane in lane_root.findall('lane'):
        lane_id = lane.get('id')
        # Safely check if the 'vehicles' tag exists
        vehicles_element = lane.find('vehicles')
        if vehicles_element is not None:
            vehicle_values = lane.find('vehicles').get('value')
            # Count the number of vehicles in this lane (number of IDs in the value)
            num_vehicles = len(vehicle_values.split()) if vehicle_values else 0
        else:
            num_vehicles = 0  # different lane tag
        lane_vehicle_count[lane_id] = num_vehicles

    return lane_vehicle_count

# Function to sum the vehicles for each junction and save to CSV
def match_junctions_to_lanes(junction_data, lane_vehicle_count, csv_filename, state_file, run_number, write_header=False):
    fieldnames = ['Junction ID', 'Incoming Lanes', 'Total Vehicle Count', 'State File', 'Run Number']
    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for junction in junction_data:
            junction_id = junction['junction_id']
            inc_lanes = junction['inc_lanes']
            total_vehicle_count = 0

            # Sum the vehicle counts for all incoming lanes
            for lane_id in inc_lanes:
                if lane_id in lane_vehicle_count:
                    total_vehicle_count += lane_vehicle_count[lane_id]

            # Write data to csv
            writer.writerow({
                'Junction ID': junction_id,
                'Incoming Lanes': ', '.join(inc_lanes),
                'Total Vehicle Count': total_vehicle_count,
                'State File' : state_file,
                'Run Number' : run_number
            })


def process_state_files(state_folder, net_file, dataSet, cars_per_intersection_file, i, density):
    """
    Processes state files for a single simulation run in parallel
    """
    InputData = []
    OutputData = []
    statistics_index = 0

    for file in state_folder.iterdir():
        statistics_location = f"data/training_data/statistics/statistics_{i}/stats_" + str(statistics_index) + ".xml"
        statistics_location = Path(os.path.abspath(statistics_location))
        print(statistics_location)

        # statistics_location = "data/training_data/statistics/stats" + str(statistics_index) + ".xml"

        if file.is_file():
            run_short_simulation_from_state(500, dataSet, file, statistics_location)
            vehicle_list = list(get_vehicles_per_road(file, net_file).values())

            # Extract data from both files
            junction_data = extract_junction_lanes(net_file)
            lane_vehicle_count = extract_lanes_with_vehicles(file)
            stateFile_name = os.path.basename(file)

            # Sum the vehicle count for each intersection
            match_junctions_to_lanes(junction_data, lane_vehicle_count, cars_per_intersection_file, stateFile_name, i, write_header=True)

            # If the sum of vehicles is greater than the density threshold, append data to InputData
            if sum(vehicle_list) >= density:
                InputData.append(get_curr_traffic_light_config(net_file) + vehicle_list)
                OutputData.append(get_waiting_time_from_statistics(statistics_location))

        statistics_index += 1
    return InputData, OutputData


def run_simulation(input_tuple):
    """
    Runs one simulation and processes state files
    """

    i, num_vehicles, density, cars_per_intersection_file = input_tuple

    start_time = time.time()

    # Set dataset number to keep consistent naming
    dataSet = str(i)
    net_file = f"data/training_data/updated_map_{dataSet}.net.xml"

    # TODO: Check that net_file does not exist.

    # randomize map to generate random traffic light timing
    # generate_random_traffic_lights("Data/map.net.xml", net_file)
    # No random traffic
    copy_file("Data/map.net.xml", net_file)

    # Set output and route files
    output_file = f"data/training_data/set{dataSet}_random_trips.rou.xml"
    output_trips_file = f"data/training_data/set{dataSet}_trips.xml"

    # TODO: Check that output_file and output_trips_file does not exist.

    # Generate random trips - 15 for low density and 25 for high density-50 for extrapolation
    generate_random_traffic(net_file, output_file, output_trips_file, num_vehicles, 15)

    # Generate SUMO config file
    configFile = f"data/training_data/set{dataSet}_data.sumocfg"
    generate_sumo_cfg(f"updated_map_{dataSet}.net.xml", f"set{dataSet}_random_trips.rou.xml", configFile)

    # Run long simulation
    state_folder = run_long_simulation(num_vehicles - 500, dataSet)

    # Process state files (this can also be parallelized within each run)
    state_folder_path = Path(os.path.abspath(state_folder))
    input_data, output_data = process_state_files(state_folder_path, net_file, dataSet, cars_per_intersection_file, i, density)

    exec_time = time.time() - start_time

    timeData = exec_time
    
    return  (i, input_data, output_data, timeData)

if __name__ == "__main__":
    num_vehicles = 2000
    density = 200
    nRuns = 1
    cars_per_intersection_file = 'Data/training_data/intersection_lane_vehiclesP.csv'


    results = []
    inputs = [(i, num_vehicles, density, cars_per_intersection_file) for i in range(nRuns)]
    # Use a Pool to parallelize the simulation runs
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        # Create tasks for each simulation run
        results = pool.map(run_simulation, inputs)

    # results is a list like this: [(i, input_data, timeData)]
    print(len(results))

    InputData = []
    OutputData = []
    timeDatas = []
    for (i, input_data, output_data, timeData) in results:
        InputData.extend(input_data)
        OutputData.extend(output_data)
        timeDatas.append([timeData])

    # Write the results to CSV after all processes have finished
    write_to_csv(InputData, "Data/training_data/inputDenseP.csv")
    write_to_csv(OutputData, "Data/training_data/outputDenseP.csv")
    write_to_csv(timeDatas, "Data/training_data/timeDataP.csv")
