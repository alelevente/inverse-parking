import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from xml.dom import minidom

DAYS = 10 #number of days to generate
ROOT = "../../02_simulation/random_grid/"

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def read_trips_df(file_path):
    trips = pd.read_xml(file_path, xpath="trip")
    trips = trips[~trips["id"].str.startswith("randUni")]
    trips["veh_id"] = trips["id"].transform(lambda x: x.split(":")[0])
    trips["move_id"] = trips["id"].transform(lambda x: int(x.split(":")[1]))
    return trips

def get_next_depart_time(trip_df, veh_id, move_id):
    moves = trip_df[trip_df["veh_id"] == veh_id]
    #as trip_df is ordered by moving times, we search for the first move_id which is greater than actual:
    for i,r in moves.iterrows():
        if r["move_id"] > move_id:
            return r["depart"]
    return -1

def get_stop_duration(trip_df, veh_id, move_id):
    moves = trip_df[trip_df["veh_id"] == veh_id]
    #as trip_df is ordered by moving times, we search for the first move_id which is greater than actual:
    start = 0
    for i,r in moves.iterrows():
        if r["move_id"] > move_id:
            return r["depart"]-start
        elif r["move_id"] == move_id:
            start = r["depart"]
    return -1

def save(file_path):
    with open(file_path, "w") as f:
        f.write(prettify(trips_with_parking_tree))

def main():
    trips_tree = ET.parse(ROOT+"gen_activities.trips.xml")
    trips_with_parking_tree = ET.Element("routes")
    processed_households = set() #one household car shall be processed only once _at all_
    trips = read_trips_df(ROOT+"gen_activities.trips.xml")

    for day in trange(DAYS):
        processed_commuters = set() #one commuter car shall be processed only once _per day_
        for elem in trips_tree.getroot():
            if elem.tag != "trip":
                if day == 0: trips_with_parking_tree.insert(-1, elem)
            else:
                trip_id = elem.get("id")
                veh_id, trip_no = trip_id.split(":")

                if veh_id.startswith("randUni"):
                    #simply adding uniform random traffic: (for each day)
                    trip_type = elem.get("type")
                    trip_depart_new = float(elem.get("depart"))+day*24*60*60 #departure on the next day
                    trip_depart_pos = elem.get("departPos")
                    trip_arrival_pos = elem.get("arrivalPos")
                    trip_arrival_speed = elem.get("arrivalSpeed")
                    trip_from = elem.get("from")
                    trip_to = elem.get("to")

                    duplicate_move = ET.SubElement(trips_with_parking_tree, "trip")
                    duplicate_move.set("id", "%s:%d"%(veh_id, day))
                    duplicate_move.set("type", trip_type)
                    duplicate_move.set("depart", str(trip_depart_new))
                    duplicate_move.set("departPos", trip_depart_pos)
                    duplicate_move.set("arrivalPos", trip_arrival_pos)
                    duplicate_move.set("arrivalSpeed", trip_arrival_speed)
                    duplicate_move.set("from", trip_from)
                    duplicate_move.set("to", trip_to)

                if (veh_id.startswith("h")) and (day == 0) and (not(veh_id in processed_households)):
                    #households:
                    trip_with_stops = generate_stops_households(trips, veh_id, days=DAYS)
                    trips_with_parking_tree.insert(len(trips_with_parking_tree), trip_with_stops)
                    processed_households.add(veh_id)

                if (veh_id.startswith("carIn")) and (not(veh_id in processed_commuters)):
                    #commuters:
                    commuter_stops = generate_stops_commuters(trips, veh_id, day)
                    trips_with_parking_tree.insert(len(trips_with_parking_tree), commuter_stops)
                    processed_commuters.add(veh_id)
                    
    save(ROOT+"burnin_trips.trip.xml")
                    
if __name__ == "__main__":
    main()