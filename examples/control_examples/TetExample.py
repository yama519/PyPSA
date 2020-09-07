<<<<<<< HEAD
from __future__ import print_function, division
import pandas as pd
import pypsa
import numpy as np
from pypsa.descriptors import get_switchable_as_iter

snapshot = pd.date_range(start="2020-01-01 00:00", end="2020-01-01 12:00",  periods=1)
L = pd.Series([-1]*1, snapshot)

n = pypsa.Network()
n.set_snapshots(snapshot)
n.add("Bus", "MV bus", v_nom=20, v_mag_pu_set=1.02)
n.add("Bus", "LV1 bus", v_nom=.4)
n.add("Bus", "LV2 bus", v_nom=.4)
#n.add("Bus", "LV3 bus", v_nom=.4)
n.add("Transformer", "MV-LV trafo", type="0.63 MVA 10/0.4 kV", bus0="MV bus", bus1="LV1 bus", tap_side=1, tap_position=-1, oltc=True)
n.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
n.add("Generator", "External Grid", bus="MV bus", control="Slack")
n.add("Load", "LV load", bus="LV1 bus", p_set=1, s_nom=1.5, power_factor_min=0.9, control_strategy = '')
n.add("Load", "test load", bus="LV1 bus", p_set=0, s_nom=1.3)
n.add("Load", "test load2", bus="LV2 bus", p_set=1, s_nom=1.2, power_factor=0.1, control_strategy = '')  
n.add("Load", "test load3", bus="LV1 bus", p_set=1, s_nom=1.3, power_factor=0.3, control_strategy = '')
n.add("Generator", "PV", bus="LV2 bus", control="PQ", p_set =1, q_set = 0, s_nom=1.5)

n.add("Load", "test load4", bus="LV1 bus", p_set=1, q_set=1)
n.add("Store", "store2", bus="LV2 bus", p_set=0,q_set=0)
n.add("StorageUnit", "store2", bus="LV2 bus", p_set=0, control_strategy = '', power_factor=0.90)
n.add("StorageUnit", "store1", bus="LV2 bus", q_set=0, p_set=-L, control_strategy = '')

n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots, inverter_control= True)  # False  # True





=======
# from __future__ import print_function, division
# import os
# import sys
# import pandas as pd
# import pypsa
# import numpy as np

# from pypsa.descriptors import get_switchable_as_iter
# git_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import_name = os.path.join(git_folder, 'Otput')
# snapshot = pd.date_range(start="2020-01-01 00:00", end="2020-01-01 12:00",  periods=1)
# L = pd.Series([-1]*1, snapshot)

# n = pypsa.Network()
# n.set_snapshots(snapshot)
# n.add("Bus", "MV bus", v_nom=20, v_mag_pu_set=1.02)
# n.add("Bus", "LV1 bus", v_nom=.4)
# n.add("Bus", "LV2 bus", v_nom=.4)
# #n.add("Bus", "LV3 bus", v_nom=.4)
# n.add("Transformer", "MV-LV trafo", type="0.63 MVA 10/0.4 kV", bus0="MV bus", bus1="LV1 bus", tap_side=1, oltc=True, deadband = 2)
# n.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
# n.add("Generator", "External Grid", bus="MV bus", control="Slack")
# n.add("Load", "LV load", bus="LV1 bus", p_set=1, s_nom=1.5, damper=0.5, control_strategy = '')
# n.add("Load", "test load", bus="LV1 bus", p_set=0, s_nom=1.3)
# n.add("Load", "test load2", bus="LV2 bus", p_set=1, s_nom=1.2, power_factor=0.95, control_strategy = '')  
# n.add("Load", "test load3", bus="LV1 bus", p_set=1, s_nom=1.3, power_factor=0.95, damper = 0.7, control_strategy = '')
# n.add("Generator", "PV1", bus="LV2 bus", control="PQ", p_set=1, damper=0.2, v_pu_cr=1.02, s_nom=1.5, power_factor=0.9, control_strategy = '')
# n.add("Generator", "PV2", bus="LV1 bus", control="PQ", p_set=1, s_nom=1.5, power_factor=0.9, control_strategy = 'q_v')
# n.add("Load", "test load4", bus="LV1 bus", p_set=1, q_set=1)
# n.add("Store", "store2", bus="LV2 bus", p_set=0,q_set=0)
# n.add("StorageUnit", "store2", bus="LV2 bus", p_set=0, control_strategy = '', power_factor=0.90)
# n.add("StorageUnit", "store1", bus="LV2 bus", q_set=0, p_set=-L, control_strategy = '')

# n.lpf(n.snapshots)
# n.pf(use_seed=True, snapshots=n.snapshots, inverter_control = False, oltc_control = False)  # False  # True


# df = n.generators
>>>>>>> 90e8e9101fc84913dea56454e077572d31ec3269
# def prepare_controlled_index_dict(n, inverter_control, snapshots, oltc_control):

#     # p_set = get_switchable_as_dense(n, c, 'p_set', snapshots, c_attrs.index)
#     n_trials_max = 0
#     parameter_dict = {}
#     ctrl_list = ['', 'q_v', 'p_v', 'cosphi_p', 'fixed_cosphi']
#     if oltc_control:
#         if (n.transformers.oltc).any():
#             parameter_dict['oltc'] = {}
#             parameter_dict['oltc']['Transformer'] = n.transformers.loc[(n.transformers.oltc == True)]

#     if inverter_control:
#         # loop through loads, generators, storage_units and stores if they exist
#         for c in n.iterate_components(n.controllable_one_port_components):


#                 # exclude slack generator to be controlled
#                 if c.list_name == 'generators':
#                     c.df.loc[c.df.control == 'Slack', 'control_strategy'] = ''
#                 # if voltage dep. controller exist,find the bus name
#                 n_trials_max = np.where(
#                       c.df.control_strategy.isin(['q_v', 'p_v']).any(), 30, 0)

#                 for i in ctrl_list[1:5]:
#                     # building a dictionary for each controller if they exist
#                     if (c.df.control_strategy == i).any():
#                         if i not in parameter_dict:
#                             parameter_dict[i] = {}

#                         parameter_dict[i][c.name] = c.df.loc[(
#                                 c.df.control_strategy == i)]

#     return parameter_dict

<<<<<<< HEAD
# def injection_curtailment(v_pu_bus, v_pu_cr, v_max_curtail, p_inj):
    
#     pperpmax = np.select([(v_pu_bus < v_pu_cr), (v_pu_bus > v_max_curtail), (
#         (v_pu_bus >= v_pu_cr) & (v_pu_bus <= v_max_curtail))], [
#             100, 0, (100-(100/(v_max_curtail-v_pu_cr))*(v_pu_bus-v_pu_cr))])
            
#     p_out = ((pperpmax*(p_inj)) / 100)
    
#     return p_out


# injection_curtailment(1.03303598, 1.02, 1.04, 0.06804)


# tap_step=2.5
# meas_min = 0.955205
# meas_max = 1.042984

# taps = np.arange(-2, 3)
# current_tap = 2
# # taps = taps[np.where(taps!=current_tap)]
# v_min=0.95
# v_max=1.02
# tap_max = 2
# tap_min = -2
# max_voltage = meas_max-taps*tap_step/100 + current_tap*tap_step/100
# min_voltage = meas_min-taps*tap_step/100 + current_tap*tap_step/100

# opt_ind = np.where(((min_voltage > v_min) & (max_voltage < v_max)))[0]

# if len(opt_ind) != 0:
#     opt_tap = taps[opt_ind[0]]
# else:
#     opt_ind = np.where(min_voltage > v_min)[0]
#     if len(opt_ind) != 0:
#         opt_tap = taps[len(opt_ind)-1]
#     else:
#         opt_tap = taps[0]

# # print('optimum tap position',opt_tap)

# max_voltage = meas_max + (current_tap - taps)*tap_step/100
# max_voltage = meas_max-taps*tap_step/100 + current_tap*tap_step/100

# # if opt_tap > tap_max or opt_tap < tap_min:
# #     opt_tap = np.select([opt_tap > tap_max, opt_tap < tap_min],
# #                         [tap_max, tap_min])
# current_tap = 0
# # with tap 2 it is 0.91, with tap 1 it is 0.935
# (0.96 + (2-current_tap)*(-1)*tap_step/100)

# (v_diff.max() > x_tol_outer or oltc) and n_trials < n_trials_max:
=======
# dict_controlled_index = prepare_controlled_index_dict(n, True, snapshot, True)
# import collections
# dict_controlled_index = collections.OrderedDict(sorted(dict_controlled_index.items()))

# for control in dict_controlled_index.keys():
#     print('controllllll', control)
>>>>>>> 90e8e9101fc84913dea56454e077572d31ec3269
    
# tap_changed = None
# # opt_tap = -1
# # c = -1
# # cur_tap = 2
# # 0.91 + (opt_tap*c-cur_tap)*c*2.5/100


# tap_step=2.5
# meas_min = 0.955205
# meas_max = 1.042984

# taps = np.arange(-2, 3)
# current_tap = 2
# # taps = taps[np.where(taps!=current_tap)]
# v_min=0.95
# v_max=1.02
# tap_max = 2
# tap_min = -2
# max_voltage = meas_max-taps*tap_step/100 + current_tap*tap_step/100
# min_voltage = meas_min-taps*tap_step/100 + current_tap*tap_step/100

# opt_ind = np.where(((min_voltage > v_min) & (max_voltage < v_max)))[0]

# if len(opt_ind) != 0:
#     opt_tap = taps[opt_ind[0]]
# else:
#     opt_ind = np.where(min_voltage > v_min)[0]
#     if len(opt_ind) != 0:
#         opt_tap = taps[len(opt_ind)-1]
#     else:
#         opt_tap = taps[0]

# print('optimum tap position',opt_tap)

# max_voltage = meas_max + (current_tap - taps)*tap_step/100
# max_voltage = meas_max-taps*tap_step/100 + current_tap*tap_step/100

# # if opt_tap > tap_max or opt_tap < tap_min:
# #     opt_tap = np.select([opt_tap > tap_max, opt_tap < tap_min],
# #                         [tap_max, tap_min])
# current_tap = 0
# # with tap 2 it is 0.91, with tap 1 it is 0.935
# (0.96 + (2-current_tap)*(-1)*tap_step/100)

# (v_diff.max() > x_tol_outer or oltc) and n_trials < n_trials_max:
    
### single bus control
# v_set = 1
# v_pu_ctrl_buses = 1.05
# tap_position = 0
# taps = np.arange(-2, 3)
# tap_step = 2.5
# possible_tap_res = abs(v_set-v_pu_ctrl_buses -
#                 (tap_position - taps)*tap_step/100*v_set)

# opt_tap = taps[np.where(possible_tap_res == min(possible_tap_res))][0] 
# main function of oltc

    # n_iter_oltc += 1
    # tap_changed = False
    # for ind in index:
        
    #     row = n.transformers.loc[ind]
    #     # c is contant to consider tap side issue in the end
    #     c = find_oltc_tap_side(n, row)

    #     current_tap = row['tap_position']
    #     # extracting controlled nodes names
    #     buses = [x.strip() for x in n.transformers.loc[ind, 'ctrl_buses'].split(',')]

    #     # if no node is chosen take node of secondary of trafo as control node
    #     ctrl_buses = np.where(
    #             len(n.transformers.loc[ind, 'ctrl_buses']) == 0, [row['bus1']], buses)
    #     # find voltages of controlled nodes
    #     v_pu_ctrl_buses = n.buses_t.v_mag_pu.loc[snapshot, ctrl_buses]
        
    #     # Single node oltc control part:
    #     if len(ctrl_buses) == 1:
    #         opt_tap, tap_step = trafo_single_node_ctrl(
    #                              n, snapshot, index, ind, row, v_pu_ctrl_buses)

    #     # Multiple node oltc control part:
    #     elif len(ctrl_buses) > 1:
    #         opt_tap, tap_step = trafo_multiple_bus_ctrl(
    #                              n, snapshot, index, ind, row, v_pu_ctrl_buses)

    #     # set the optimum tap position calculated either from single or multiple
    #     # node, and recalculte admittance matrix.

    #     n.transformers_t.opt_tap_position.loc[snapshot, ind] = opt_tap
    #     print('voltage', v_pu_ctrl_buses, 'opt_tap', opt_tap, 'snap', i, 'trial', n_trials)
    #     ## copy from trafo types for referece
    #     # t["tap_ratio"] = 1. + (t["tap_position"] - t["tap_neutral"]) * (t["tap_step"]/100.)
    #     if current_tap != opt_tap:
    #         print('opt', opt_tap, 'curt', current_tap)
    #         tap_changed = True
    #         n.transformers.loc[ind, 'tap_position'] = opt_tap
    #         ratio = (row['tap_ratio'] + (opt_tap - current_tap)*c*tap_step/100)
    #         n.transformers.loc[ind, 'tap_ratio'] = ratio

    #         # TODO I will dig into "calculate_Y" to extract only the needed part after after my thesis since it needs more work.
    #         calculate_Y(sub_network, skip_pre=skip_pre)
    #     # if n_iter_oltc > 2:
    #     #     tap_changed = np.where(v_pu_ctrl_buses.min() >= row['v_min'], True, False)
    #     print('check if tap is changed or not, tap_chanded?', tap_changed, 'opt', opt_tap, 'curt', current_tap)
    # return tap_changed, n_iter_oltc

# trafor multiple bus
# def trafo_multiple_bus_ctrl(n, snapshot, index, ind, row, v_pu_ctrl_buses):
#     current_tap = row['tap_position']
#     taps, tap_step = find_taps_tap_steps(n, snapshot, index, row, ind)
#     opt_tap = 0
#     # exclusion of current tap from the tap options
#     taps = taps[np.where(taps!=current_tap)]
#     meas_max = v_pu_ctrl_buses.values.max()
#     meas_min = v_pu_ctrl_buses.values.min()

#     # check if meas_max and meas_min are withing the range
#     if (meas_min > row['v_min'] and meas_max < row['v_max']):
#         logger.info(" Voltage in nodes %s controlled by oltc in  %s are"
#                     " already withing 'v_min' and 'v_max' ranges.",
#                     v_pu_ctrl_buses.index.tolist(), ind)
#         pass

#     # if they are not withing the range then find optimum tap as follow:
#     else:
#         max_voltage = meas_max-(taps-abs(current_tap))*tap_step*row['v_set']/100
#         min_voltage = meas_min-(taps-abs(current_tap))*tap_step*row['v_set']/100

#         opt_ind = np.where(((min_voltage > row['v_min']) & (
#                 max_voltage < row['v_max'])))[0]

#         if len(opt_ind) != 0:
#             opt_tap = taps[opt_ind[0]]
#         else:
#             opt_ind = np.where(min_voltage > row['v_min'])[0]
#             if len(opt_ind) != 0:
#                 opt_tap = taps[len(opt_ind)-1]

#             else:
#                 opt_tap = taps[0]
    
#     opt_tap = current_tap + opt_tap
#     if opt_tap > row['tap_max'] or opt_tap < row['tap_min']:
#         opt_tap = np.select([opt_tap > row['tap_max'], opt_tap < row['tap_min']],
#                             [row['tap_max'], row['tap_min']])

#         logger.info("The voltage in %s controlled by oltc in %s, using "
#                     " %s as the optimum tap position.",
#                     v_pu_ctrl_buses.index.tolist(), ind, opt_tap)

#     return opt_tap, tap_step



        # set the optimum tap position calculated from single or multiple
# if n_iter_oltc > 5:
    
#     logger.info("Due to unconsistancy of oltc parameters in %s transformer"
#                 "with multiple bus control, oltc is jumping around the two"
#                 " tap positions %s and %s which oscilates around lower"
#                 "voltage limit '%s'. By default, oltc is in favour of "
#                 "avoiding under voltage problems, thus it chooses %s as"
#                 " the optimum tap position. If this is undesirable please "
#                 "change the lower voltage limit 'v_min' to avoid the problem ",
#                 ind, current_tap, opt_tap, row['v_min'], current_tap)
    
    # opt_tap = np.where(
    #         v_pu_ctrl_buses.min() > row['v_min'], current_tap, opt_tap)

<<<<<<< HEAD
    
=======

##########    SIMULATION ENVIRONMENT    ##########
# important for exporting the data
    # Result_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # export = os.path.join(Result_folder, 'Graphics', control_type)
    
# calculate the nember of generators
# all_cur_pv_plants = n.generators.loc[n.generators.control == 'PQ']
# num_of_pv_plants = len(all_cur_pv_plants)
# occupied_buses = np.unique(all_cur_pv_plants.bus.values)
# all_grid_buses = n.buses.loc[n.buses.control == 'PQ'].index


# available_buses = all_grid_buses.difference(occupied_buses)
# num_of_new_pv_plants_scen_30 = round(num_of_pv_plants*1.3 - num_of_pv_plants)
# num_of_new_pv_plants_scen_173 = round(num_of_pv_plants*2.73 - num_of_pv_plants)

# network.madd("Generator", ["gen 1", "load 2"], bus=["1","2"], p_set=np.random.rand(len(network.snapshots),2))


# import time
# def is_prime(n):
#       if (n <= 1) : 
#           return 'not a prime number'
#       if (n <= 3) : 
#           return 'prime number'
          
#       if (n % 2 == 0 or n % 3 == 0) : 
#           return 'not a prime number'
    
#       i = 5
#       while(i * i <= n) : 
#           if (n % i == 0 or n % (i + 2) == 0) : 
#               return 'not a prime number'
#           i = i + 6
    
#       return 'prime number'
    
# starttime = time.time()
# for i in range(1,10):
#     time.sleep(2)
#     print('{} is {} number'.format(i, is_prime(i)))
# print()    
# print('Time taken = {} seconds'.format(time.time() - starttime))    

































>>>>>>> 90e8e9101fc84913dea56454e077572d31ec3269
