import argparse
import re

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def read_trips_df(file_path, random_regex):
    def not_filter_regex(text):
        '''Returns *True* if the regex cannot be matched to the text,
         else otherwise.'''
        
        if re.findall(random_regex, file_path):
            return False
        else:
            return True
        
    trips = pd.read_xml(file_path, xpath="trip")
    trips = trips[trips["id"].apply(not_filter_regex)]
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

def generate_stops_households(trip_df, veh_id, days=1):
    '''
        For households, this function replaces trips by stops at the given destinations.
    '''
    moves = trip_df[trip_df["veh_id"] == veh_id]
    home = moves.iloc[0]
    
    new_trip = ET.Element("trip")
    new_trip.set("id", "%s:0"%(veh_id))
    new_trip.set("type", "default")
    new_trip.set("depart", str(home["depart"]))
    new_trip.set("departPos", str(home["departPos"]))
    new_trip.set("arrivalPos", str(home["arrivalPos"]))
    new_trip.set("from", str(home["from"]))
    new_trip.set("to", str(home["from"]))
    
    for day in range(days):
        for i,r in moves.iterrows():
            new_stop = ET.SubElement(new_trip, "stop")
            new_stop.set("parkingArea", "pa%s"%r["to"])
            
            move_id = int(r["move_id"])
            stop_duration = get_stop_duration(moves, veh_id, move_id)
            if stop_duration == -1: #last trip
                stop_duration = np.random.normal(home["depart"]+24*60*60 - r["depart"], 7*60, 1)[0] #staying at home for a random time
                
            new_stop.set("duration", str(stop_duration))
            
    return new_trip

def generate_stops_commuters(trip_df, veh_id, day):
    moves = trip_df[trip_df["veh_id"] == veh_id]
    home = moves.iloc[0]
    out = moves.iloc[-1]
    
    departure_time = float(home["depart"]) + 24*60*60*day
    new_trip = ET.Element("trip")
    new_trip.set("id", "%s:%d"%(veh_id, day))
    new_trip.set("type", "default")
    new_trip.set("depart", str(departure_time))
    new_trip.set("departPos", str(home["departPos"]))
    new_trip.set("arrivalPos", str(home["arrivalPos"]))
    new_trip.set("from", str(home["from"]))
    new_trip.set("to", str(out["to"]))
    
    #adding parking during the day:
    move_id = home["move_id"]
    parking_time = get_stop_duration(moves, veh_id, move_id)
    new_stop = ET.SubElement(new_trip, "stop")
    new_stop.set("duration", str(parking_time))
    new_stop.set("parkingArea", "pa%s"%home["to"])
    
    return new_trip

def save(file_path, trips_with_parking_tree):
    with open(file_path, "w") as f:
        f.write(prettify(trips_with_parking_tree))

def main(args):
    trips_tree = ET.parse(args.input)
    trips_with_parking_tree = ET.Element("routes")
    processed_households = set() #one household car shall be processed only once _at all_
    trips = read_trips_df(args.input, args.random_regex)

    for day in range(args.days):
        processed_commuters = set() #one commuter car shall be processed only once _per day_
        for elem in trips_tree.getroot():
            if elem.tag != "trip":
                if day == 0: trips_with_parking_tree.insert(-1, elem)
            else:
                trip_id = elem.get("id")
                veh_id, trip_no = trip_id.split(":")

                if re.findall(args.random_regex, veh_id):
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

                if (day == 0) and (re.findall(args.household_regex, veh_id)) and (not(veh_id in processed_households)):
                    #households:
                    #Household traffic (due to its continuity during the simulated time) will be handled
                    #on the first day. This will generate all the moves of the household vehicles for
                    #all the simulated days:
                    trip_with_stops = generate_stops_households(trips, veh_id, days=args.days)
                    trips_with_parking_tree.insert(len(trips_with_parking_tree), trip_with_stops)
                    processed_households.add(veh_id)

                if (re.findall(args.commuter_regex, veh_id)) and (not(veh_id in processed_commuters)):
                    #commuters:
                    #Commuters (as they enter and leave the simulation regularly) can be
                    #handeled iteratively for each day.
                    commuter_stops = generate_stops_commuters(trips, veh_id, day)
                    trips_with_parking_tree.insert(len(trips_with_parking_tree), commuter_stops)
                    processed_commuters.add(veh_id)
                    
    save(args.output, trips_with_parking_tree)
                    
if __name__ == "__main__":
    #parsing input arguments:
    parser = argparse.ArgumentParser(
                prog="Parking Activities",
                description='''Adds parking to activities (generated by activitygen). Vehicle
                               ids are required to end with ":X", where X distinguishes the
                               trips of the vehicle during a day, similarly to activitygen.''')
    
    #mandatory arguments:
    parser.add_argument("input", help="input trip file")
    parser.add_argument("output", help="output trip file with parking")
    
    #optional arguments:
    parser.add_argument("--days",
                        help="how many days to simulate",
                        nargs="?",
                        default=5,
                        type=int)
    
    parser.add_argument("--random_regex",
                        help="regular expression for random traffic",
                        nargs="?",
                        default="randUni[\d]*")
    
    parser.add_argument("--household_regex",
                        help="regular expression for household traffic",
                        nargs="?",
                        default="h[\d]*c[\d]*")
    
    parser.add_argument("--commuter_regex",
                        help="regular expression for commuter traffic",
                        nargs="?",
                        default="carIn[\d]*")
    
    args = parser.parse_args() #processing arguments
    
    main(args)