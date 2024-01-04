import xml.etree.ElementTree as ET

if __name__=="__main__":
    #read the parking lot definition file:
    parking_lot_tree = ET.parse("../02_scenario/parking_areas.add.xml")

    #doubling capacities:
    for pa in parking_lot_tree.getroot().findall('parkingArea'):
        capacity = int(pa.get("roadsideCapacity"))
        capacity *= 2
        pa.set("roadsideCapacity", str(capacity))
        
    #saving the results:
    parking_lot_tree.write("../02_scenario/parking_areas.add.xml")    
