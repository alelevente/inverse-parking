#!/bin/bash

netgenerate --rand -o ../02_scenario/rand_grid.net.xml --rand.iterations=30 --rand.grid --tls.guess --roundabouts.guess -R --tls.default-type actuated --tls.set 1 --no-turnarounds.except-deadend

activitygen -n ../02_scenario/rand_grid.net.xml -s statistics.xml -o ../02_scenario/gen_activities.trips.xml


#generating parking lots:
$SUMO_HOME/tools/generateParkingAreas.py -n ../02_scenario/rand_grid.net.xml -a 90 -o ../02_scenario/parking_areas.add.xml
python3 ./double_capacities.py
python3 ./add_parking_lots.py #adds garages
$SUMO_HOME/tools/generateParkingAreaRerouters.py -n ../02_scenario/rand_grid.net.xml -a ../02_scenario/parking_areas.add.xml -o ../02_scenario/parking_rerouters.add.xml --max-distance-alternatives 750
python3 ./extend_rerouter_timings.py #extends rerouter timing to 11 days

duarouter -n ../02_scenario/rand_grid.net.xml -r ../02_scenario/gen_activities.trips.xml -o ../02_scenario/gen_activities.rou.xml --ignore-errors -a ../02_scenario/parking_areas.add.xml

python3 ./parking_activities.py
