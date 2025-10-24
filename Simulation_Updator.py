import xml.etree.ElementTree as Et

'''Small script to update the ID's of vehicles in a .rou file to combine two route files in a single SUMO config'''
if __name__ == "__main__":
    tree1 = Et.parse("setHD.rou.xml")
    root1 = tree1.getroot()

    tree2 = Et.parse("setLD.rou.xml")
    root2 = tree2.getroot()

    initial_vehicle_count = int(root2.findall('.//vehicle')[-1].get("id"))+1
    vehicles_to_append = root1.findall('.//vehicle')
    for vehicle in vehicles_to_append:
        vehicle.set("id", str(int(vehicle.get("id"))+initial_vehicle_count))

    tree1.write("setHD.rou.xml")

