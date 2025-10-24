import csv
import subprocess
import random
import sys
import xml.etree.ElementTree as Et
from pathlib import Path

import time
import os

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


def generate_random_traffic_lights(input_net_file, output_net_file):
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
    traci.start([checkBinary('sumo'), "-c", "data/training_data/set" + set_number + "_data.sumocfg", "--start",
                 "--quit-on-end", "--save-state.period", "100", "--save-state.suffix", ".xml",
                 "--save-state.prefix",
                 "data/training_data/states/"])
    time = traci.simulation.getTime()
    # run the simulation until 500 time steps before the last vehicle spawn
    run(time + duration)


def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


    
def run_short_simulation_from_state(duration, set_number, state, statistics_file):
    """Run a simulation for a shorter period to generate a result based on a given state"""
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

if __name__ == "__main__":
    num_vehicles = 2000
    InputData = []
    resultData = []
    timeData = []
    density = 200

    
    nRuns = 600
    for i in range(nRuns):
        start_time = time.time()
        # set dataset number to keep consistent naming
        dataSet = str(i)
        # set net file location
        net_file = "data/training_data/updated_map_" + dataSet + ".net.xml"

        # randomize map to generate random traffic light timing
        #generate_random_traffic_lights("Data/map.net.xml", net_file)
        copy_file("Data/map.net.xml", net_file)
        # set output and route and trips files
        output_file = "data/training_data/" + "set" + dataSet + "_random_trips.rou.xml"
        output_trips_file = "data/training_data/set" + dataSet + "_trips.xml"

        # set config file location
        configFile = "data/training_data/set" + dataSet + "_data.sumocfg"
        # generate random trips for the randomized map
        generate_random_traffic(net_file, output_file, output_trips_file, num_vehicles, 25)
        # generate sumo config file to run
        generate_sumo_cfg("updated_map_" + dataSet + ".net.xml", "set" + dataSet + "_random_trips.rou.xml",
                          configFile)

        # run simulation and save the state every 100 time steps, define the location of output files
        run_long_simulation(num_vehicles - 500, dataSet)

        state_folder = Path("Data/training_data/states")
        traffic_lights = get_curr_traffic_light_config(net_file)
        cars_per_intersection_file = 'Data/training_data/intersection_lane_vehicles.csv'
        # run the simulation for 500 time steps for every save state
        statistics_index = 0
        for file in state_folder.iterdir():
            statistics_location = "data/training_data/statistics/stats" + str(statistics_index) + ".xml"
            if file.is_file():
                #I dont need this for checking uniformity
                #run_short_simulation_from_state(500, dataSet, file, statistics_location)
                vehicle_list = list(get_vehicles_per_road(file, net_file).values())

                # Extract data from both files
                junction_data = extract_junction_lanes(net_file)
                lane_vehicle_count = extract_lanes_with_vehicles(file)
                stateFile_name = os.path.basename(file)
                # number of cars per intersection
                match_junctions_to_lanes(junction_data, lane_vehicle_count, cars_per_intersection_file, stateFile_name, i, write_header=True)


                if sum(vehicle_list) >= density:
                    InputData.append(get_curr_traffic_light_config(net_file) + vehicle_list)
                    #resultData.append(get_waiting_time_from_statistics(statistics_location))
                # remove the state so that we don't clog our folder with unnecessary files
                # file.unlink()

                
            statistics_index += 1
        exec_time = time.time() - start_time                
        timeData.append([(i),exec_time])
        write_to_csv(InputData, "Data/training_data/inputDense2.csv")
        write_to_csv(resultData, "Data/training_data/outputDense2.csv")
        write_to_csv(timeData, "Data/training_data/timeData.csv")
