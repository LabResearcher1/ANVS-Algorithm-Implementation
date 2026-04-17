import datetime
import itertools
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import CG_Algorithm as mf
from gams import *
import time
import os
import sys



def main():
    inst_id = 3
    term_name = 'small'
    case_id = 1
    # folder =  'Realworld_instances'
    folder = 'synthetic_instance'
    subfolder = r'instance_' + str(inst_id)
    # subfolder = term_name + '_testnetwork'
    # the path of the files storing nodes, links and travel costs
    # node_file = 'complete_nodes.csv'
    node_file = 'complete_nodes.csv'

    travel_data_file = 'complete_travel_data.csv'
    # travel_data_file = 'complete_travel_data.csv'

    service_time_file = 'case5_service_time.csv'
    # service_time_file = 'base_service_time.csv'

    waste_demand_file = 'waste_demand.csv'


    print(node_file)
    print(travel_data_file)
    print(service_time_file)
    print(waste_demand_file)



    # mf.solve(os.path.join(folder, subfolder, node_file),
    #          os.path.join(folder, subfolder, travel_data_file),
    #          os.path.join(folder, subfolder, service_time_file),
    #          os.path.join(folder, subfolder, waste_demand_file))

    mf.solve(os.path.join(folder, subfolder, node_file), \
              os.path.join(folder, subfolder, travel_data_file), os.path.join(folder, subfolder, service_time_file),
              os.path.join(folder, subfolder, waste_demand_file))


if __name__ == "__main__":
    main()


# def main():
#     # inst_id = 3
#     # term_name = 'small'
#     # case_id = 1
#     # folder =  'shortterm_network'
#     folder = 'augmented_routes_for_charging'
#     # subfolder = r'instance_' + str(inst_id)
#     # the path of the files storing nodes, links and travel costs
#     # node_file = 'complete_nodes.csv'
#     node_file = 'route_3_aug_nodes.csv'
#
#     travel_data_file = 'route_3_aug_travel.csv'
#     # travel_data_file = 'complete_travel_data.csv'
#
#     # service_time_file = 'case_service_time.csv'
#     service_time_file = 'route_3_aug_service_time.csv'
#
#     # waste_demand_file = 'waste_demand.csv'
#     waste_demand_file = 'route_3_aug_waste_demand.csv'
#
#     print(node_file)
#     print(travel_data_file)
#     print(service_time_file)
#     print(waste_demand_file)
#
#
#     # mf.solve(os.path.join(folder, subfolder, node_file),
#     #          os.path.join(folder, subfolder, travel_data_file),
#     #          os.path.join(folder, subfolder, service_time_file),
#     #          os.path.join(folder, subfolder, waste_demand_file))
#
#     # mf.solve(os.path.join(folder, subfolder, node_file), \
#     #           os.path.join(folder, subfolder, travel_data_file), os.path.join(folder, subfolder, service_time_file),
#     #           os.path.join(folder, subfolder, waste_demand_file))
#
#     mf.solve(os.path.join(folder, node_file), \
#               os.path.join(folder, travel_data_file), os.path.join(folder, service_time_file),
#               os.path.join(folder, waste_demand_file))


# if __name__ == "__main__":
#     main()








