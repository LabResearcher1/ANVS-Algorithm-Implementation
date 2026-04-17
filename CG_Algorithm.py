import datetime
import math
from itertools import combinations
from sklearn.cluster import KMeans
import itertools
import pickle
import networkx as nx
import numpy as np
from gams import *
import pandas as pd
import CG_Algorithm as mf
import mip_solve as exact
import rmip_solve as rmip
import operator
import os
import re
import logging
import sys
import datetime
import itertools
from gams import GamsWorkspace, GamsExceptionExecution
import warnings


warnings.filterwarnings("ignore")


def compute_shortest_path(df):

    # Create graphs for distance and time
    G_distance = nx.DiGraph()  # Directed graph for distance
    G_time = nx.DiGraph()  # Directed graph for time

    # Add edges to graphs
    for _, row in df.iterrows():
        from_node = 'i' + str(int(row['from_node ID']))
        to_node = 'i' + str(int(row['to_node ID']))
        distance = float(row['Distance (miles)'])
        time = float(row['Time (minutes)'])

        # Add edges with weights
        G_distance.add_edge(from_node, to_node, weight=distance)
        G_time.add_edge(from_node, to_node, weight=time)

    # Compute all-pairs shortest paths
    shortest_paths_distance = dict(nx.all_pairs_dijkstra_path_length(G_distance, weight='weight'))
    shortest_paths_time = dict(nx.all_pairs_dijkstra_path_length(G_time, weight='weight'))

    travel_dist_dict = {}
    travel_time_dict = {}

    for source, targets in shortest_paths_distance.items():
        travel_dist_dict[source] = targets

    for source, targets in shortest_paths_time.items():
        travel_time_dict[source] = targets

    return travel_dist_dict, travel_time_dict


# def determine_charge_time(route):
#
#     remaining_range = max_driving_range  # Initialize with maximum range
#     full_charge_time = 0  # Initialize the full charge time to zero
#
#     for idx in range(len(route) - 1):
#         # Identify the current and next nodes
#         cur_id = route[idx][0]
#         if cur_id in truck_info.keys():
#             cur_node = truck_info[cur_id]['loc']
#         elif cur_id in community_info.keys():
#             cur_node = community_info[cur_id]['loc']
#         else:  # Charging station
#             cur_node = charge_station_info[cur_id]["off_charge_node"]
#
#         next_id = route[idx + 1][0]
#         if next_id in truck_info.keys():
#             next_node = truck_info[next_id]['loc']
#         elif next_id in community_info.keys():
#             next_node = community_info[next_id]['loc']
#         else:  # Charging station
#             next_node = charge_station_info[next_id]["off_charge_node"]
#
#         # If the next node is a charging station, calculate the charging time needed
#         if next_id in charge_station_info.keys():
#             charge_time = route[idx + 1][1]
#             on_charge_node = charge_station_info[next_id]["on_charge_node"]
#             off_charge_node = charge_station_info[next_id]["off_charge_node"]
#             distance = travel_dist_dict[cur_node][on_charge_node] + travel_dist_dict[on_charge_node][off_charge_node]
#
#             # Update remaining range after traveling to and from the charging station
#             remaining_range -= distance
#             # Calculate the required time to fully charge
#             full_charge_time = (max_driving_range - remaining_range) / charge_rate
#             # Reset remaining range to maximum after charging
#             remaining_range = max_driving_range
#         else:
#             # Update remaining range for a regular node
#             distance = travel_dist_dict[cur_node][next_node]
#             remaining_range -= distance
#
#     # Calculate redundant distance after completing the route
#     redundant_dist = remaining_range - min_driving_range
#
#     # Adjust the charging time considering the redundant distance
#     if full_charge_time > 0:
#         charge_time = full_charge_time - (redundant_dist / charge_rate)
#     else:
#         charge_time = 0
#
#     # Ensure charge time respects the minimum charge time requirement
#     if charge_time > 0:
#         return True, max(min_charge_time, charge_time)
#     else:
#         # No charging needed, return 0
#         return False, 0


def determine_charge_time(route):

    remaining_range = max_driving_range

    for idx in range(len(route) - 1):

        cur_id = route[idx][0]
        if cur_id in truck_info:
            cur_node = truck_info[cur_id]['loc']
        elif cur_id in community_info:
            cur_node = community_info[cur_id]['loc']
        else:
            cur_node = charge_station_info[cur_id]["off_charge_node"]

        next_id = route[idx + 1][0]

        # ---------------- CHARGING STATION ----------------
        if next_id in charge_station_info:

            on_node = charge_station_info[next_id]["on_charge_node"]
            off_node = charge_station_info[next_id]["off_charge_node"]

            # distance to reach charging station
            dist_to_cs = (
                travel_dist_dict[cur_node][on_node] +
                travel_dist_dict[on_node][off_node]
            )

            remaining_range -= dist_to_cs

            # 🔥 compute required future distance
            future_dist = 0
            temp_node = off_node

            for j in range(idx + 1, len(route) - 1):

                next_j_id = route[j + 1][0]

                if next_j_id in truck_info:
                    next_j_node = truck_info[next_j_id]['loc']
                elif next_j_id in community_info:
                    next_j_node = community_info[next_j_id]['loc']
                else:
                    next_j_node = charge_station_info[next_j_id]["off_charge_node"]

                future_dist += travel_dist_dict[temp_node][next_j_node]
                temp_node = next_j_node

            # 🔥 include safety buffer
            required_energy = future_dist + min_driving_range

            # 🔥 compute how much to charge
            needed_charge = max(0, required_energy - remaining_range)

            charge_time = needed_charge / charge_rate

            if charge_time > 0:
                charge_time = max(min_charge_time, charge_time)
                return True, charge_time

        # ---------------- NORMAL ARC ----------------
        else:
            if next_id in truck_info:
                next_node = truck_info[next_id]['loc']
            elif next_id in community_info:
                next_node = community_info[next_id]['loc']
            else:
                next_node = charge_station_info[next_id]["off_charge_node"]

            remaining_range -= travel_dist_dict[cur_node][next_node]

    return False, 0


def insert_charge_station(route):

    feasible_inserts = []
    for cs_id in charge_station_info.keys():

        for i in range(1, len(route)):
        # Try inserting at start, middle, and near end

            cur_insert = route[:i] + [(cs_id,0)] + route[i:]
            # print(f"route_with_cs: {cur_insert}")
            # print(f"route_loc with charge: {mf.get_route_loc(cur_insert)}")

            charge_feasible, charge_time = mf.determine_charge_time(cur_insert)
            # print(f"if_feasible:{charge_feasible}, charge_time: {charge_time}")

            if charge_feasible:
                cur_insert = route[:i] + [(cs_id, charge_time)] + route[i:]
                #if cur_insert not in route_pool:
                is_feasible = mf.check_route_feasibility(cur_insert)
                if is_feasible == True:
                    # print(f"really feasible: {cur_insert}")
                    route_pool.append(cur_insert)
                    feasible_inserts.append(cur_insert)


    return feasible_inserts



def check_route_feasibility(route):
    load = 0
    remaining_range = max_driving_range

    for idx in range(len(route) - 1):
        cur_id = route[idx][0]

        if cur_id in truck_info.keys():
            cur_node = truck_info[cur_id]['loc']
        elif cur_id in community_info.keys():
            cur_node = community_info[cur_id]['loc']
        else:
            cur_node = charge_station_info[cur_id]["off_charge_node"]

        next_id = route[idx + 1][0]
        if next_id in truck_info.keys():
            next_node = truck_info[next_id]['loc']
        elif next_id in community_info.keys():
            next_node = community_info[next_id]['loc']
        else:
            next_node = charge_station_info[next_id]["off_charge_node"]

        if next_id in community_info.keys():
            load = load + community_info[next_id]['waste_amount']
            if load > truck_cap:
                return 'capacity_violation'

        if next_id in charge_station_info.keys():
            charge_time = route[idx + 1][1]
            on_charge_node = charge_station_info[next_id]["on_charge_node"]
            off_charge_node = charge_station_info[next_id]["off_charge_node"]
            distance = travel_dist_dict[cur_node][on_charge_node] + travel_dist_dict[on_charge_node][off_charge_node]

            remaining_range = remaining_range - distance
            range_increment = charge_time * charge_rate
            remaining_range = min(max_driving_range, remaining_range + range_increment)
        else:
            distance = travel_dist_dict[cur_node][next_node]
            remaining_range = remaining_range - distance

            if remaining_range < min_driving_range - 0.001:
                return 'range_violation'

    return True


def generate_1v1(truck_id, cm_id):
    return [(truck_id, 0), (cm_id, community_info[cm_id]["service_time"]), (truck_id, 0)]


def get_basic_solution(gams_job):
    sol = []
    for rec in gams_job.out_db["x"]:
        if rec.level > 0:
            sol.append(rec.keys[0])
    return sol



def generate_best_route_by_one_insertion(route, am_id_insert, community_dual):

    list_good_routes = []
    # print(f"cur rte for insertion: {route}, am_id_insert: {am_id_insert}")

    for i in range(1, len(route)):
        # Create a new list with the new node inserted at the i-th position
        # print(f"route[:i] and route[i:]: {route[:i]}, {route[i:]}")
        cur_insert = route[:i] + [(am_id_insert, community_info[am_id_insert]['service_time'])] + route[i:]
        # print(f"insert: {cur_insert}")

        # Check if the new route is feasible
        if cur_insert not in route_pool:
            if_feasible = mf.check_route_feasibility(cur_insert)

            if if_feasible == True:
                #if cur_insert not in route_pool:

                cm_ids = [id[0] for id in cur_insert if id[0] in community_info.keys()]
                # minimizing problem: find negative reduced cost, i.e, compute reduced cost
                total_travel_dist, total_travel_time, charging_time = mf.find_route_dist_time(cur_insert)
                #total_travel_dist, total_travel_time = mf.find_route_dist_time(cur_insert)
                # total_route_cost = total_travel_dist*unit_dist_cost + total_travel_time*unit_time_cost
                total_route_cost = total_travel_dist * unit_dist_cost + total_travel_time * unit_time_cost + charging_time * unit_charge_cost
                reduced_cost = total_route_cost - sum(community_dual[cm] for cm in cm_ids)

                # Add feasible route and its reduced cost
                if reduced_cost < -0.00001:
                    list_good_routes.append(cur_insert)
                    route_pool.append(cur_insert)
            else:
            # the infeasibility is resulted by exceeding maximum driving range
                if if_feasible == 'range_violation':

                    fesible_routes_with_charge = mf.insert_charge_station(cur_insert)
                    if len(fesible_routes_with_charge) > 0:

                        for fesible_route in fesible_routes_with_charge:
                            if fesible_route not in route_pool:

                                cm_ids = [id[0] for id in fesible_route if id[0] in community_info.keys()]
                                # maximizing problem: find positive reduced cost
                                #total_travel_dist, total_travel_time = mf.find_route_dist_time(fesible_route)
                                total_travel_dist, total_travel_time, charging_time = mf.find_route_dist_time(cur_insert)
                                # total_route_cost = total_travel_dist*unit_dist_cost + total_travel_time*unit_time_cost
                                total_route_cost = total_travel_dist * unit_dist_cost + total_travel_time * unit_time_cost + charging_time * unit_charge_cost
                                reduced_cost = total_route_cost - sum(community_dual[cm] for cm in cm_ids)

                                if reduced_cost < -0.00001:
                                    list_good_routes.append(fesible_route)
                                    route_pool.append(fesible_route)

    if len(list_good_routes) > 0:
        return True, list_good_routes
    else:
        return False, None


###############################################################################################################################
#
# def generate_best_route_by_one_insertion(route, am_id_insert, community_dual):
#     list_good_routes = []
#     dict_reduced_cost = {}
#
#     # Iterate over possible insertion points
#     for i in range(1, len(route)):
#
#     # positions = {1, len(route) // 2, len(route) - 1}
#     # for i in sorted(positions):
#     # for i in range(1, len(route), 2):
#
#         # Create a new list with the new node inserted at the i-th position
#         cur_insert = route[:i] + [(am_id_insert, community_info[am_id_insert]['service_time'])] + route[i:]
#
#         # Convert the route to a tuple for hashable storage
#         cur_insert_tuple = tuple(cur_insert)
#
#         # Check if the new route is feasible
#         if cur_insert_tuple not in route_pool:
#             if_feasible = mf.check_route_feasibility(cur_insert)
#
#             if if_feasible == True:
#                 # Compute reduced cost
#                 cm_ids = [id[0] for id in cur_insert if id[0] in community_info.keys()]
#                 total_travel_dist, total_travel_time, charging_time = mf.find_route_dist_time(cur_insert)
#                 #total_travel_dist, total_travel_time = mf.find_route_dist_time(cur_insert)
#                 total_route_cost = total_travel_dist * unit_dist_cost + total_travel_time * unit_time_cost + charging_time*unit_charge_cost
#                 #total_route_cost = total_travel_dist * unit_dist_cost + total_travel_time * unit_time_cost
#                 reduced_cost = total_route_cost - sum(community_dual[cm] for cm in cm_ids)
#
#
#                 # Add feasible route and its reduced cost
#                 if reduced_cost < -0.00001:
#                     route_pool.append(cur_insert_tuple)
#                     list_good_routes.append(cur_insert)
#                     dict_reduced_cost[cur_insert_tuple] = reduced_cost
#
#
#             elif if_feasible == 'range_violation':
#                 # Handle range violations by inserting charging stations
#                 fesible_routes_with_charge = mf.insert_charge_station(cur_insert)
#                 for fesible_route in fesible_routes_with_charge:
#                     fesible_route_tuple = tuple(fesible_route)  # Convert to tuple
#                     if fesible_route_tuple not in route_pool:
#                         cm_ids = [id[0] for id in fesible_route if id[0] in community_info.keys()]
#                         total_travel_dist, total_travel_time, charging_time = mf.find_route_dist_time(fesible_route)
#                         #total_travel_dist, total_travel_time = mf.find_route_dist_time(fesible_route)
#                         total_route_cost = total_travel_dist * unit_dist_cost + total_travel_time * unit_time_cost + charging_time*unit_charge_cost
#                         #total_route_cost = total_travel_dist * unit_dist_cost + total_travel_time * unit_time_cost
#                         reduced_cost = total_route_cost - sum(community_dual[cm] for cm in cm_ids)
#
#
#                         route_pool.append(fesible_route_tuple)
#                         list_good_routes.append(fesible_route)
#                         dict_reduced_cost[fesible_route_tuple] = reduced_cost
#                         #print(f"Route after charging station insertion: {fesible_route}, Reduced Cost: {reduced_cost}")
#
#     # Sort routes by reduced cost
#     if list_good_routes:
#         list_good_routes = sorted(list_good_routes, key=lambda x: dict_reduced_cost[tuple(x)])
#
#         # Find the index of the first positive reduced cost
#         positive_index = next((i for i, route in enumerate(list_good_routes) if dict_reduced_cost[tuple(route)] >= 0), len(list_good_routes))
#
#         # Calculate the 50% index
#         percentage_index = int(len(list_good_routes) * 0.2)
#
#         # Retain routes with reduced cost below the threshold
#         list_good_routes = list_good_routes[:max(percentage_index, positive_index)]
#
#     # Return the results
#     if list_good_routes:
#         return True, list_good_routes
#     else:
#         return False, None

#####################################################################################################################################


def get_route_loc(route):
    route_loc = []
    for idx in route:
        idx = idx[0]
        if idx in truck_info.keys():
            route_loc.append(truck_info[idx]['loc'])
        elif idx in community_info.keys():
            route_loc.append(community_info[idx]['loc'])
        else:
            on_charge_node = charge_station_info[idx]["on_charge_node"]
            off_charge_node = charge_station_info[idx]["off_charge_node"]
            route_loc.extend([on_charge_node, off_charge_node])

    return route_loc


# def find_route_dist_time(route):
#     total_travel_dist = 0
#     total_travel_time = 0
#
#     for idx in range(len(route) - 1):
#         cur_id = route[idx][0]
#
#         if cur_id in truck_info.keys():
#             cur_node = truck_info[cur_id]['loc']
#         elif cur_id in community_info.keys():
#             cur_node = community_info[cur_id]['loc']
#         else:
#             cur_node = charge_station_info[cur_id]["off_charge_node"]
#
#         next_id = route[idx + 1][0]
#         if next_id in truck_info.keys():
#             next_node = truck_info[next_id]['loc']
#         elif next_id in community_info.keys():
#             next_node = community_info[next_id]['loc']
#         else:
#             next_node = charge_station_info[next_id]["off_charge_node"]
#
#         total_travel_time += route[idx + 1][1]
#
#         if next_id in charge_station_info.keys():
#             on_charge_node = charge_station_info[next_id]["on_charge_node"]
#             off_charge_node = charge_station_info[next_id]["off_charge_node"]
#             total_travel_dist += travel_dist_dict[cur_node][on_charge_node] + travel_dist_dict[on_charge_node][
#                 off_charge_node]
#             total_travel_time += travel_time_dict[cur_node][on_charge_node] + travel_time_dict[on_charge_node][
#                 off_charge_node]
#         else:
#             total_travel_dist += travel_dist_dict[cur_node][next_node]
#             total_travel_time += travel_time_dict[cur_node][next_node]
#
#     return total_travel_dist, total_travel_time



def find_route_dist_time(route):
    total_travel_dist  = 0
    total_travel_time  = 0   # driving + service
    total_charge_time  = 0   # charging only

    for idx in range(len(route) - 1):
        cur_id  = route[idx][0]
        nxt_id  = route[idx+1][0]
        t       = route[idx+1][1]  # service time or charging time at that stop

        # figure out the actual node names
        if cur_id in truck_info:
            cur_node = truck_info[cur_id]['loc']
        elif cur_id in community_info:
            cur_node = community_info[cur_id]['loc']
        else:
            # it's a charging‐off node
            cur_node = charge_station_info[cur_id]['off_charge_node']

        if nxt_id in truck_info:
            nxt_node = truck_info[nxt_id]['loc']
        elif nxt_id in community_info:
            nxt_node = community_info[nxt_id]['loc']
        else:
            # it's a charging‐off node
            nxt_node = charge_station_info[nxt_id]['off_charge_node']

        # always add the time spent at that node (service or charge)
        # but we'll separate out charging from driving+service in the totals below
        if nxt_id in charge_station_info:
            # driving to station + station→off‐node
            on  = charge_station_info[nxt_id]['on_charge_node']
            off = charge_station_info[nxt_id]['off_charge_node']

            total_travel_dist += (
                travel_dist_dict[cur_node][on]
              + travel_dist_dict[on][off]
            )
            total_travel_time += (
                travel_time_dict[cur_node][on]
              + travel_time_dict[on][off]
            )
            total_charge_time += t

        else:
            # normal driving leg
            total_travel_dist += travel_dist_dict[cur_node][nxt_node]
            total_travel_time += travel_time_dict[cur_node][nxt_node]
            total_travel_time += t  # add service time here

    return total_travel_dist, total_travel_time, total_charge_time




# def add_to_all_route(route, if_route_feasible, truck_id):
#
#     route_id = f"r{len(all_routes) + 1}"
#
#     all_routes[route_id] = {}
#     all_routes[route_id]["route_id"] = route_id
#     all_routes[route_id]["route"] = route
#     all_routes[route_id]["route_loc"] = mf.get_route_loc(route)
#     all_routes[route_id]["isFeasible"] = if_route_feasible
#     total_travel_dist, total_travel_time = find_route_dist_time(route)
#     all_routes[route_id]["trip_milage"] = total_travel_dist
#     all_routes[route_id]["trip_duration"] = total_travel_time
#     all_routes[route_id]["truck_id"] = truck_id
#     all_routes[route_id]["cs_ids"] = [id[0] for id in route if id[0] in charge_station_info.keys()]
#     cm_covered = [id[0] for id in route if id[0] in community_info.keys()]
#     all_routes[route_id]["cm_ids"] = cm_covered
#     all_routes[route_id]["count_cm"] = len(cm_covered)
#
#     total_load = 0
#     total_service_time = 0
#     total_charge_time = 0
#     for cm in cm_covered:
#         total_load = total_load + community_info[cm]['waste_amount']
#         total_service_time = total_service_time + community_info[cm]['service_time']
#
#     for idx in range(len(route)):
#         cur_id = route[idx][0]
#         if cur_id in charge_station_info.keys():
#             total_charge_time = total_charge_time + route[idx][1]
#
#
#     all_routes[route_id]["waste_collection"] = total_load
#     all_routes[route_id]["service_time"] = total_service_time
#     all_routes[route_id]["charge_time"] = total_charge_time
#
#     all_routes[route_id]["route_cost"] = unit_dist_cost*total_travel_dist + unit_time_cost*total_travel_time
#
#
#     return route_id



def add_to_all_route(route, if_route_feasible, truck_id):
    route_id = f"r{len(all_routes) + 1}"

    all_routes[route_id] = {}
    all_routes[route_id]["route_id"]     = route_id
    all_routes[route_id]["route"]        = route
    all_routes[route_id]["route_loc"]    = mf.get_route_loc(route)
    all_routes[route_id]["isFeasible"]   = if_route_feasible

    # <-- unpack the three values now
    total_travel_dist, total_travel_time, total_charging_time = find_route_dist_time(route)

    all_routes[route_id]["trip_milage"]   = total_travel_dist
    all_routes[route_id]["trip_duration"] = total_travel_time
    all_routes[route_id]["truck_id"]      = truck_id

    all_routes[route_id]["cs_ids"] = [
        nid for nid, _ in route if nid in charge_station_info
    ]
    cm_covered = [
        nid for nid, _ in route if nid in community_info
    ]
    all_routes[route_id]["cm_ids"]    = cm_covered
    all_routes[route_id]["count_cm"]  = len(cm_covered)

    # compute total load & service time
    total_load = sum(community_info[cm]['waste_amount']   for cm in cm_covered)
    total_service_time = sum(community_info[cm]['service_time'] for cm in cm_covered)

    all_routes[route_id]["waste_collection"] = total_load
    all_routes[route_id]["service_time"]     = total_service_time
    all_routes[route_id]["charge_time"]      = total_charging_time

    # <-- updated cost to include charging
    all_routes[route_id]["route_cost"] = (
         unit_dist_cost   * total_travel_dist
       + unit_time_cost   * total_travel_time
       + unit_charge_cost * total_charging_time
    )

    return route_id


# Function for generating 2-community routes
def generate_nvm_routes(m):

    # print("generate_nvm_routes = ", m)

    nvm_route_ids = {}
    prem_route_ids = all_route_ids[m - 1]


    for truck_id in truck_info.keys():
        # check the route ids covered by currect truck
        cur_truck_prem_route_ids = prem_route_ids[truck_id]
        cur_truck_vm_route_ids = []
        for cur_truck_rtd in cur_truck_prem_route_ids:
            route = all_routes[cur_truck_rtd]['route']
            cm_covered = all_routes[cur_truck_rtd]['cm_ids']
            for cm_id in community_info.keys():
                if cm_id not in cm_covered:
                    for i in range(1, len(route)):
                        cur_insert = route[:i] + [(cm_id, community_info[cm_id]['service_time'])] + route[i:]
                        # print(f"route: {route}, cur_insert_route:{cur_insert}")
                        if cur_insert not in route_pool:
                            if_feasible = mf.check_route_feasibility(cur_insert)
                            #print(f"if feasible: {if_feasible}, route_loc:{mf.get_route_loc(cur_insert)}")
                            if if_feasible == True:
                                route_pool.append(cur_insert)
                                route_id = mf.add_to_all_route(cur_insert, True, truck_id)
                                cur_truck_vm_route_ids.append(route_id)
                            else:
                                if if_feasible == 'range_violation':
                                    # print(f"try to insert charge station..")
                                    fesible_routes_with_charge = mf.insert_charge_station(cur_insert)
                                    # print(f"try to insert charge station..")
                                    # mf.pause_and_continue()
                                    if len(fesible_routes_with_charge) > 0:
                                        # print(f"find the route with charge station: {fesible_routes_with_charge}")
                                        for fesible_route in fesible_routes_with_charge:
                                            route_pool.append(fesible_route)
                                            route_id = mf.add_to_all_route(fesible_route, True, truck_id)
                                            cur_truck_vm_route_ids.append(route_id)

        nvm_route_ids[truck_id] = cur_truck_vm_route_ids

    all_route_ids[m] = nvm_route_ids
    print(f"m: {m}; # number of routes: {len(all_routes)}")



def solve(node_file_path, travel_data_file_path, service_time_file_path, waste_demand_file_path):

    global travel_dist_dict, travel_time_dict, max_driving_range, min_driving_range, community_info, charge_station_info, truck_info, truck_cap, all_route_ids, all_routes, route_pool
    global unit_dist_cost, unit_time_cost, unit_charge_cost, charge_rate, min_charge_time
    unit_dist_cost = 2.0
    unit_time_cost = 0.6  # 0.35
    unit_charge_cost = 0.8 # 1.56
    charge_rate = 0.5 # mile per minute
    min_charge_time = 5 # 20
    ###################### varying values of parameters across different instance #################
    max_driving_range = 45 # 150
    min_driving_range = 5 # 30
    fleet_size = 6 # 1
    truck_cap = 28000 # 26000
    ################################################################################################

    max_cg_iter = 30
    insert_strategy = 'IS#1'

    # init_max_cm_count = min(3, truck_cap)
    init_max_cm_count = 2

    # compute travel distance and travel time for complete graph
    travel_data = pd.read_csv(travel_data_file_path)
    node_data = pd.read_csv(node_file_path)
    service_time_data = pd.read_csv(service_time_file_path)
    waste_demand_data = pd.read_csv(waste_demand_file_path)


    travel_dist_dict, travel_time_dict = mf.compute_shortest_path(travel_data)


    community_info = dict()
    truck_info = dict()
    charge_station_info = dict()

    # for idx in range(len(node_data)):
    #     node_loc = node_data.loc[idx, 'Node ID']
    #     if node_data.loc[idx, 'Label'] == 'V':
    #         service_time = service_time_data.loc[service_time_data['Node'] == node_loc, 'Service Time'].iloc[0]
    #         # Read waste amount from the same CSV (adjust the column name if necessary)
    #         waste_amount = waste_demand_data.loc[waste_demand_data['Node'] == node_loc, 'Demand'].iloc[0]
    #         community_info[f"cm{len(community_info) + 1}"] = {
    #             "cm_id": f"cm{len(community_info) + 1}",
    #             "loc": "i" + str(node_data.loc[idx, 'Node ID']),
    #             "waste_amount": float(waste_amount),
    #             "service_time": float(service_time)
    #         }
    #     if node_data.loc[idx, 'Label'] == 'D':
    #         truck_info[f"truck{len(truck_info) + 1}"] = {
    #             "truck_id": f"truck{len(truck_info) + 1}",
    #             "loc": "i" + str(node_data.loc[idx, 'Node ID']),
    #         }
    #
    #     if node_data.loc[idx, 'Label'] == 'C+':
    #         on_charge_node = node_data.loc[idx, 'Node ID']
    #         off_charge_node = int(
    #             travel_data.loc[travel_data['from_node ID'] == on_charge_node, 'to_node ID'].values[0])
    #         charge_station_info[f"cs{len(charge_station_info) + 1}"] = {
    #             "cs_id": f"cs{len(charge_station_info) + 1}",
    #             "on_charge_node": "i" + str(on_charge_node),
    #             "off_charge_node": "i" + str(off_charge_node)
    #         }

    def normalize_node_id(x):
        return str(int(float(x)))

    for idx in range(len(node_data)):

        raw_node_id = node_data.loc[idx, 'Node ID']
        node_id = normalize_node_id(raw_node_id)  # ← FIXED
        node_loc = "i" + node_id

        label = node_data.loc[idx, 'Label']

        # ------------------ COMMUNITY (V) ------------------
        if label == 'V':

            service_time = service_time_data.loc[
                service_time_data['Node'].astype(float) == float(raw_node_id),
                'Service Time'
            ].iloc[0]

            waste_amount = waste_demand_data.loc[
                waste_demand_data['Node'].astype(float) == float(raw_node_id),
                'Demand'
            ].iloc[0]

            community_info[f"cm{len(community_info) + 1}"] = {
                "cm_id": f"cm{len(community_info) + 1}",
                "loc": node_loc,  # ← FIXED
                "waste_amount": float(waste_amount),
                "service_time": float(service_time)
            }

        # ------------------ DEPOT (D) ------------------
        elif label == 'D':

            truck_info[f"truck{len(truck_info) + 1}"] = {
                "truck_id": f"truck{len(truck_info) + 1}",
                "loc": node_loc  # ← FIXED
            }

        # ------------------ CHARGING STATION (C+) ------------------
        elif label == 'C+':

            on_charge_node = node_id  # ← FIXED

            # Find matching row robustly (float-safe)
            off_node_raw = travel_data.loc[
                travel_data['from_node ID'].astype(float) == float(raw_node_id),
                'to_node ID'
            ].values[0]

            off_charge_node = normalize_node_id(off_node_raw)  # ← FIXED

            charge_station_info[f"cs{len(charge_station_info) + 1}"] = {
                "cs_id": f"cs{len(charge_station_info) + 1}",
                "on_charge_node": "i" + on_charge_node,  # ← FIXED
                "off_charge_node": "i" + off_charge_node  # ← FIXED
            }

    print(f"community_info is {community_info}")
    print(f"truck_info is {truck_info}")
    print(f"charge_station_info is {charge_station_info}")

    # Time_1 = datetime.datetime.now()


    all_route_ids = {}   # save the route ids for each truck that service varying number of customers
    all_routes = {}  # save all generated routes
    route_pool = []

    # generate all nv1 routes (initial feasible routes)
    nv1_route_ids = {}
    for truck_id in truck_info.keys():
        cur_truck_route_ids = []
        for cm_id in community_info.keys():

            # Generate a route with a single truck serving a single community
            route = mf.generate_1v1(truck_id, cm_id)
            # print(route)
            # mf.pause_and_continue()

            # Check if the route is feasible
            if_route_feasible = mf.check_route_feasibility(route)
            if if_route_feasible == True:
                # If feasible, add to route pool and record it
                if route not in route_pool:
                    route_pool.append(route)
                    route_id = mf.add_to_all_route(route, if_route_feasible, truck_id)
                    cur_truck_route_ids.append(route_id)
            else:
                # If not feasible, check if inserting charging stations makes it feasible
                if if_route_feasible == 'range_violation':   # check if the route needs to visit a charging station
                    fesible_routes_with_charge = mf.insert_charge_station(route)
                    if len(fesible_routes_with_charge) > 0:
                        for fesible_route in fesible_routes_with_charge:
                            #if fesible_route not in route_pool:
                            route_id = mf.add_to_all_route(fesible_route, True, truck_id)
                            cur_truck_route_ids.append(route_id)

        nv1_route_ids[truck_id] = cur_truck_route_ids

    # print(f"m: {1}; # number of routes: {len(all_routes)}")
    all_route_ids[1] = nv1_route_ids

    # print(f"1 cover:{all_routes}")



    for m in range(2, init_max_cm_count + 1):
        generate_nvm_routes(m)


    select_num = len(all_routes)

    # # Print the final set of initial columns
    print("\nInitial columns (1-community, 2-community, etc.) generated:")
    for route_id, route_data in all_routes.items():
        print(f"Route ID: {route_id}, Route: {route_data['route']}")


    all_keys = list(all_routes.keys())
    select_keys = all_keys[:select_num]


    updated_routes = dict()
    updated_routes = {k: all_routes[k] for k in select_keys}



    # all_route_ids = {}  # save the route ids for each truck that service varying number of customers
    # all_routes = {}  # save all generated routes
    # route_pool = []  # Pool of all generated routes
    #
    # # # Generate and add 1-community routes
    # nv1_route_ids = {}
    # for truck_id in truck_info.keys():
    #     cur_truck_route_ids = []
    #     for cm_id in community_info.keys():
    #         # print(f"Processing Truck: {truck_id}, Community: {cm_id}")
    #
    #         # Generate a route with a single truck serving a single community
    #         route = mf.generate_1v1(truck_id, cm_id)
    #         # print(f"Generated Route: {route}")
    #
    #         # Check if the route is feasible
    #         if_route_feasible = mf.check_route_feasibility(route)
    #         # print(f"Feasibility of Route: {if_route_feasible}")
    #
    #         if if_route_feasible == True:
    #             # If feasible, add to route pool and record it
    #             if route not in route_pool:
    #                 route_pool.append(route)
    #                 route_id = mf.add_to_all_route(route, if_route_feasible, truck_id)
    #                 cur_truck_route_ids.append(route_id)
    #                 # print(f"Route Added: {route}")
    #         else:
    #             # If not feasible, check if inserting charging stations makes it feasible
    #             if if_route_feasible == 'range_violation':
    #                 print(f"Route violates range, attempting to insert charging stations: {route}")
    #                 fesible_routes_with_charge = mf.insert_charge_station(route)
    #                 if len(fesible_routes_with_charge) > 0:
    #                     for fesible_route in fesible_routes_with_charge:
    #                         if fesible_route not in route_pool:
    #                             route_pool.append(fesible_route)
    #                             route_id = mf.add_to_all_route(fesible_route, True, truck_id)
    #                             cur_truck_route_ids.append(route_id)
    #                             # print(f"Route Added After Charging Station Insertion: {fesible_route}")
    #                 else:
    #                     print(f"Route still infeasible even after attempting charging station insertion: {route}")
    #
    #     nv1_route_ids[truck_id] = cur_truck_route_ids
    #
    # # Store the 1-community routes
    # # print(f"m: {1}; # number of routes: {len(all_routes)}")
    # all_route_ids[1] = nv1_route_ids
    #
    # # Generate and add 2-community and 3-community routes to the initial columns
    # for m in range(2, init_max_cm_count + 1):
    #     generate_nvm_routes(m)
    #
    # # Combine all generated routes (1-community, 2-community, etc.)
    # for m in range(2, init_max_cm_count + 1):
    #     all_route_ids[1] = {**all_route_ids[1], **all_route_ids[m]}
    #
    # # Print the final set of initial columns
    # print("\nInitial columns (1-community, 2-community, etc.) generated:")
    # for route_id, route_data in all_routes.items():
    #     print(f"Route ID: {route_id}, Route: {route_data['route']}")
    #
    # select_num = len(all_routes)
    #
    # all_keys = list(all_routes.keys())
    # select_keys = all_keys[:select_num]
    #
    # updated_routes = dict()
    # updated_routes = {k: all_routes[k] for k in select_keys}



    # for key in all_routes.keys():
    #     if all_routes[key]['count_cm'] == 3:
    #         print(all_routes[key]['route'])

    # mf.pause_and_continue()

    newly_added_col_dic = {}

    if len(sys.argv) > 1:
        ws = GamsWorkspace(system_directory=r"C:\GAMS\48", debug=DebugLevel.Verbose)
    else:
        ws = GamsWorkspace(debug=DebugLevel.KeepFiles)

    db = ws.add_database()
    r_set = db.add_set("r", 1, "candidate route set")
    [r_set.add_record(route_id) for route_id in all_routes.keys()]

    a_set = db.add_set("a", 1, "community id set")
    [a_set.add_record(f"cm{i}") for i in range(1, len(community_info) + 1)]

    fleet_size_par = db.add_parameter('fleet_size', 0, 'fleet size')
    fleet_size_par.add_record().value = fleet_size

    # optimize the following codes
    route_community_par = GamsParameter(db, "route_community", 2, "route community incidence")
    route_cost_par = GamsParameter(db, "route_cost", 1, "route cost")
    for route_id, route_data in all_routes.items():
        # print(f"route_id: {route_id}, route_cm: {route_data["cm_ids"]}")
        for cm_id in route_data["cm_ids"]:
            route_community_par.add_record((route_data["route_id"], cm_id)).value = 1

        route_cost_par.add_record(route_data["route_id"]).value = route_data["route_cost"]

    Time_2 = datetime.datetime.now()

    print(f"the number of initial routes: {len(all_routes)}")
    print(f"the size of route pool: {len(route_pool)}")

    for CG_iter in range(max_cg_iter):

        promising_col_pool = []

        # print(len(route_pool))

        """ If need to check model status"""
        cp = ws.add_checkpoint()
        t3 = GamsJob(ws, source=rmip.get_model_txt_rmip())
        # t3 = GamsJob(ws, source = exact.get_model_txt_mip())
        opt = GamsOptions(ws)
        opt.defines["gdxincname"] = db.name
        opt.all_model_types = "Gurobi"  # 'Cplex'

        t3.run(opt, databases=db, checkpoint=cp)

        t3 = ws.add_job_from_string(
            "solve MyModel minimizing z using rmip; ms=MyModel.modelstat; ss=MyModel.solvestat;", cp)
        t3.run(opt, databases=db)
        if not (t3.out_db["ms"].find_record().value == 1 and t3.out_db["ss"].find_record().value == 1):
            print("\n Modelstatus: " + str(t3.out_db["ms"].find_record().value))
            print(" Solvestatus: " + str(t3.out_db["ss"].find_record().value))
            raise ValueError('RMIP is infeasible. Please enable larger cliques in the initial routes')

        rmp_solution_listBasis = mf.get_basic_solution(t3)

        for slt_rte_id in rmp_solution_listBasis:
            if slt_rte_id not in promising_col_pool:
                promising_col_pool.append(slt_rte_id)

        for slt_rte_id in rmp_solution_listBasis:
            # print(f'Iter: {CG_iter}, obj: {t3.out_db["z"][()].level:.2f}, route: {all_routes[slt_rte_id]['route']}, route loc: {all_routes[slt_rte_id]['route_loc']}, route cost: {all_routes[slt_rte_id]['route_cost']}')
            print(
                f"Iter: {CG_iter}, obj: {t3.out_db['z'][()].level:.2f}, route: {all_routes[slt_rte_id]['route']}, route loc: {all_routes[slt_rte_id]['route_loc']}, route cost: {all_routes[slt_rte_id]['route_cost']}")

        # truck_dual = {truck.keys[0]:truck.marginal for truck in t3.out_db["eq2"]}
        community_dual = {community.keys[0]: community.marginal for community in t3.out_db["eq1"]}
        #print(f"community_dual: {community_dual}")


        # mf.pause_and_continue()

        newly_added_col_dic[CG_iter] = []
        # count = 0
        # print(f"the len of promising pool: {len(promising_col_pool)}")
        if insert_strategy == 'IS#1':
            basis_pool = rmp_solution_listBasis
        elif insert_strategy == 'IS#2':
            basis_pool = updated_routes.keys()
        else:
            basis_pool = promising_col_pool

        for each_col_basis in basis_pool:
            # for each_col_basis in updated_routes.keys():
            # for each_col_basis in promising_col_pool:
            #     count = count + 1
            # print(f"each_col_basis: {each_col_basis}")
            route_info = all_routes[each_col_basis]
            existing_cm_ids = route_info["cm_ids"]
            route = route_info["route"]
            truck_id = route_info["truck_id"]



            for am_id_insert in community_info.keys():
                if am_id_insert not in existing_cm_ids:
                    ExistGoodRoute, list_good_routes = generate_best_route_by_one_insertion(route, am_id_insert, community_dual)


                    # print(f"insert_one_good_rtes: {list_good_routes}")
                    if ExistGoodRoute:
                        for new_route in list_good_routes:
                            route_id = mf.add_to_all_route(new_route, True, truck_id)
                            newly_added_col_dic[CG_iter].append(route_id)
                            promising_col_pool.append(route_id)

        if CG_iter >= 15 and len(newly_added_col_dic[
                                    CG_iter]) == 0:  # Ensure that after a few number of iterations, if no new columns are generated, stop the CG process
            print('no better routes could be found.')
            # print(f"the len of promising pool: {len(promising_col_pool)}")
            print(f"the len of all generated routes: {len(all_routes)},")
            # print(f"generated_better_count: {generated_better_count}, gams_selected_count: {gams_selected_count}")

            break
        else:
            for route_id in newly_added_col_dic[CG_iter]:
                r_set.add_record(route_id)
                route_cost_par.add_record(route_id).value = all_routes[route_id]["route_cost"]

                for cm_id in all_routes[route_id]["cm_ids"]:
                    route_community_par.add_record((route_id, cm_id)).value = 1

            updated_routes = {k: all_routes[k] for k in newly_added_col_dic[CG_iter]}

    cp = ws.add_checkpoint()
    t4 = GamsJob(ws, source=exact.get_model_txt_mip())
    opt = GamsOptions(ws)
    opt.defines["gdxincname"] = db.name
    opt.all_model_types = "Gurobi"  # 'Cplex'

    t4.run(opt, databases=db, checkpoint=cp)

    t4 = ws.add_job_from_string(
        "solve MyModel minimizing z using mip; ms=MyModel.modelstat; ss=MyModel.solvestat;", cp)
    t4.run(opt, databases=db)

    Time_3 = datetime.datetime.now()

    if not (t4.out_db["ms"].find_record().value == 1 and t4.out_db["ss"].find_record().value == 1):
        print("\n Modelstatus: " + str(t4.out_db["ms"].find_record().value))
        print(" Solvestatus: " + str(t4.out_db["ss"].find_record().value))
        raise ValueError('MIP is infeasible')

    # print("final num of routes:", len(updated_routes))

    # identify the decision variable x
    db_x = t4.out_db
    var_x = db_x.get_variable("x")

    # Loop through the values of the decision variable
    select_rte_list = []
    for rec in var_x:
        if rec.level > 0.1:
            select_rte_list.append(rec.keys[0])
            # print(f"Decision variable {rec.keys[0]}: {rec.level}")
    # print(f"select_rte_list is {select_rte_list}")

    # print("Objective profit= ", t4.out_db["z"][()].level)
    # print(f'# selected col {len(mf.get_basic_solution(t4))} {mf.get_basic_solution(t4)}')

    cg_route_list = []
    obj = 0
    for route_id in select_rte_list:
        obj = obj + all_routes[route_id]['route_cost']
        print(f"route: {all_routes[route_id]}")
        # print(f"route loc: {all_routes[route_id]['route_loc']}")
        # print(f"route cost: {all_routes[route_id]['route_cost']}")
        print("======================================================")
        # cg_route_list.append(all_routes[route_id]['route'])

    print(f"Objective profit = {obj}")
    print(f"(Model) Objective profit = ", t4.out_db["z"][()].level)
    # print(f"num. of iters used: {num_iter_used}")
    print(f"fleet size: {fleet_size}, truck_cap: {truck_cap}")
    print("# of routes:", len(select_rte_list))
    print('Total computation time', (Time_3 - Time_2).seconds)

    # print('save the routes by cg-based method')
    # solu_folder = r"./solution_data"
    # file_name = 'req_' + str(len(orders)) + '_seed_' + str(random_seed) + '_cg_routes.pickle'
    # with open(os.path.join(solu_folder, file_name), 'wb') as f:
    #     pickle.dump(cg_route_list, f)




def pause_and_continue():
    while True:
        user_input = input("Enter 'y' to run next instance: ").lower()
        if user_input == 'y':
            break
        else:
            print("Waiting for 'y' to run next instance...")
