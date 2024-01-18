import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json

import sys,os

SUMO_HOME = os.environ["SUMO_HOME"] #locating the simulator
sys.path.append(SUMO_HOME+"/tools")
import sumolib
from sumolib.visualization import helpers


class Option:
    #default options required by sumolib.visualization
    defaultWidth = 4
    defaultColor = (0.0, 0.0, 0.0, 0.0)
    linestyle = "solid"
    
    
def plot_dataset(net_file, vehicle_parkings, parking_position, title="",
                       color=(0.7, 0.0, 0.7, 0.66),
                       fig=None, ax=None):

    '''
        Plots a road network with edges colored according to a probabilistic distribution.
        Parameters:
            net_file: path to the net file
            vehicle_parking: list of parking lots measured by the vehicle
                If an edge is not in this map, it will get a default (light gray) color.
            parking_position:
                A dictionary of which keys are the parking lot names, and the values are the
                edge on which the parking lot resides.
            title: title of the produced plot
            color: color of the edges
            fig: if None then a new map is created; if it is given, then only special edges are overplot to the original fig
            ax: see fig

        Returns:
            a figure and axis object
    '''
    
    net = sumolib.net.readNet(net_file)
        
    scalar_map = None
    colors = {}
    options = Option()
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(22, 20))
        for e in parking_position.values():
            colors[e] = np.array((0.125, 0.125, 0.125, .25)) #edges are gray by default
            
    for pl in vehicle_parkings:
        edge = parking_position[pl]
        colors[edge] = color

    helpers.plotNet(net, colors, [], options)
    plt.title(title)
    plt.xlabel("position [m]")
    plt.ylabel("position [m]")
    ax.set_facecolor("lightgray")
    if not(scalar_map is None):
        plt.colorbar(scalar_map)

    return fig, ax