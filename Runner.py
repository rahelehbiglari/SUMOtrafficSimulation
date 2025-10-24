# this code opens SUMO simulation then you need to run it manually
# virtual environment: trafficdt
# save state at end-time of simulation -> I need to ask user when she wants to save state!
# Traffic Light program T0 - 90S

from __future__ import absolute_import
from __future__ import print_function

import optparse
import os
import sys
import xml.etree.ElementTree as Et
import queue
import threading
import time

from DigitalTwin import DigitalTwin

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def run(end_time):
    """execute the TraCI control loop"""
    digital_twin = DigitalTwin("Data/map.net.xml")
    step = 0
    tree = Et.parse("Data/map.net.xml")
    root = tree.getroot()
    result_queue = queue.Queue()

    while traci.simulation.getTime() < end_time:  # traci.simulation.getMinExpectedNumber() > 0:
        # by default a simulation step is 1 second.
        start_time = time.time()
        traci.simulationStep(step)
        step += 1
        if traci.simulation.getTime() % 500 == 0:
            traci.simulation.saveState('data/states/statesavedState.xml')
            thread = threading.Thread(target=digital_twin.optimize,
                                      args=('data/states/statesavedState.xml', result_queue,))
            thread.start()
            if not result_queue.empty():
                new_config = result_queue.get()
                new_program = []
                timing_index = 0
                for traffic_light in root.findall(".//tlLogic"):
                    traffic_light_id = traffic_light.get('id')
                    for phase in traffic_light.findall('phase'):
                        new_program.append(traci.trafficlight.Phase(new_config[timing_index], phase.get("state")))
                        timing_index += 1
                    curr_program = traci.trafficlight.getAllProgramLogics(traffic_light_id)
                    optimal_program = traci.trafficlight.Logic('0', 0, curr_program[0].currentPhaseIndex, new_program)
                    traci.trafficlight.setProgramLogic(traffic_light_id, optimal_program)
                    # print(traci.trafficlight.getAllProgramLogics(traffic_light_id))
                    new_program = []
        elapsed_time = time.time() - start_time
        # slow down the simulation to give more time for optimisation
        if elapsed_time < 0.2:
            time.sleep(0.2 - elapsed_time)
    # traci.simulation.saveState('data/states/statesavedState.xml')  # define when to save the state
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    optParser.add_option("--config_file", action="store", type="string", default="runner.sumocfg")
    optParser.add_option("--duration", action="store", type="int", default=8000)  # 1500 - this line defines end time
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    """
    Main runner of the project. Will start the runner.sumocfg scenario, attach the digital twin and run a simulation 
    with optimiser.
    """
    options = get_options()

    # run the simulation 10 time for a range of results
    for run_number in range(10):
        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        if options.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumoBinary, "-c", options.config_file, "--tripinfo-output",
                     "data/sumo_output/tripinfo.xml", "--statistic-output",
                     "data/sumo_output/stat" + str(run_number) + ".xml", "--start",
                     "--quit-on-end",
                     "--save-state.suffix", ".xml",
                     "--save-state.prefix", "data/states/",
                     "--tripinfo-output.write-unfinished", "True"])
        #  "data/sample.sumocfg", "--output-prefix", "TIME", "--tripinfo-output", "data/sumo_output/tripinfo.xml"])
        curr_time = traci.simulation.getTime()
        run(options.duration)

# start = time.time()
# run(options.duration)
# stop = time.time()
# print("------ Timing ------")
# print("simulation took: " + str(stop - start) + " seconds to complete")
