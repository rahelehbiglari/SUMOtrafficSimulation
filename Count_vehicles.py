import xml.etree.ElementTree as ET


def get_vehicles_per_road(state, net):
    tree_state = ET.parse(state)
    tree_net = ET.parse(net)
    root_state = tree_state.getroot()
    root_net = tree_net.getroot()

    vehicle_per_road = {}

    for edge in root_net.findall(".//edge"):
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


if __name__ == "__main__":
    vehicles_per_road = get_vehicles_per_road("data/states/2024-04-12-19-11-37statesavedState.xml",
                                              "data/map.net.xml")
    for point in vehicles_per_road:
        if vehicles_per_road[point] > 0:
            print(vehicles_per_road[point])
