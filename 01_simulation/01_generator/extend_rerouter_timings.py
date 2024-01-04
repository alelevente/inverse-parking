import xml.etree.ElementTree as ET

if __name__=="__main__":
    #read the parking rerouter definition file:
    rerouter_tree = ET.parse("../02_scenario/parking_rerouters.add.xml")
    
    #extending timings:
    for rerouter in rerouter_tree.getroot().findall('rerouter'):
        for interval in rerouter.findall("interval"):
            interval.set("end", str(11*24*60*60))
        
    #saving modifications:
    rerouter_tree.write("../02_scenario/parking_rerouters.add.xml")
