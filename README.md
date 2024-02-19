# **Remarks on Federated Learning of Parking Lot Occupancy in Eclipse SUMO**

This repository contains source codes of our research regarding the location privacy of communicating vehicles participating in a federated learning system.

## Simulation

We have created a virtual town for simulation environment. The source codes of the simulation (and the TraCI-based measurement scripts) are located in the `01_simulation` folder.

### The `parking_activities` tool:

To simulate the parking activities in the town, we developed the `parking_activities` tool that enhances the output of SUMO's `activitygen` tool.

The tool is located at [`01_simulation/01_generator/parking_activities.py`](https://github.com/alelevente/inverse-parking/blob/main/01_simulation/01_generator/parking_activities.py).

To simulate parking activities, one may run a burn-in simulation for a couple of days before the actual simulation. For usage examples see [`01_simulation/01_generator/generator.sh`](https://github.com/alelevente/inverse-parking/blob/main/01_simulation/01_generator/generator.sh), and simulations [`01_simulation/02_scenario/burnin.sumocfg`](https://github.com/alelevente/inverse-parking/blob/main/01_simulation/02_scenario/burnin.sumocfg), [`01_simulation/02_scenario/stable.sumocfg`](https://github.com/alelevente/inverse-parking/blob/main/01_simulation/02_scenario/stable.sumocfg).

## Data Analysis:

Sources codes for data analysis, including machine learning and privacy evaluation codes are located in the `02_da` folder.