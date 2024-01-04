import xml.etree.ElementTree as ET

def add_parking_lot(id_, on_lane, capacity):
    new_parking = ET.Element("parkingArea")
    new_parking.set("id", id_)
    new_parking.set("lane", on_lane)
    new_parking.set("roadsideCapacity", str(capacity))
    return new_parking

if __name__=="__main__":
    #read the parking lot definition file:
    parking_lot_tree = ET.parse("../02_scenario/parking_areas.add.xml")
    
    #adding some garages:
    p_root = parking_lot_tree.getroot()
    p_root.insert(0, add_parking_lot("pa-industrial1", "144_0", 150))
    p_root.insert(0, add_parking_lot("pa-industrial2", "-144_0", 150))
    p_root.insert(0, add_parking_lot("pa-central", "131_0", 300))
    p_root.insert(0, add_parking_lot("pa-residential_south", "-120_0", 250))
    p_root.insert(0, add_parking_lot("pa-residential_east", "-131_0", 300))
    p_root.insert(0, add_parking_lot("pa-residential_west", "162_0", 150))
    p_root.insert(0, add_parking_lot("pa-outside1", "31_0", 100))
    p_root.insert(0, add_parking_lot("pa-outside2", "-11_0", 100))
    
    #saving updated parking lot definition file:
    parking_lot_tree.write("../02_scenario/parking_areas.add.xml")
