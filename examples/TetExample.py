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
#network.add("Bus", "LV3 bus", v_nom=.4) # , type = '0.4 MVA 20/0.4 kV'
n.add("Transformer", "MV-LV trafo", bus0="MV bus", bus1="LV1 bus",
      tap_position=0, tap_ratio=1, deadband=5, tap_min=-2, tap_max=2, tap_step=2.5, v_min=0.9,  
      v_max=1.02, ctrl_nodes="LV1 bus", oltc = True, s_nom=0.4,x=0.0582833, r=0.01425, g=0.003376, tap_side=0)

n.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
n.add("Generator", "External Grid", bus="MV bus", control="Slack")
n.add("Load", "LV load", bus="LV1 bus", p_set=1, s_nom=8, power_factor_min=0.1, control_strategy = '') # , control_strategy = 'fixed_cosphi'
n.add("Load", "test load", bus="LV1 bus", p_set=1, s_nom=1.3)
n.add("Load", "test load2", bus="LV2 bus", p_set=0, power_factor=0.95, s_nom=1.3, q_set=0, power_factor_min=0.1, control_strategy = 'fixed_cosphi')  
n.add("Load", "test load3", bus="LV1 bus", p_set=0, s_nom=1.3, q_set=0, power_factor=0.3, control_strategy = '')
n.add("Generator", "PV", bus="LV2 bus", control="PQ", p_set=0, q_set=-0)
n.add("Load", "test load4", bus="LV1 bus", p_set=0, q_set=0)
#network.add("Store", "store2", bus="LV2 bus", p_set=1,q_set=0)
n.add("StorageUnit", "store2", bus="LV2 bus", p_set=0.5, p_nom=1, inflow=0, control_strategy = 'p_v', power_factor=0.95)
n.add("StorageUnit", "store1", bus="LV2 bus", q_set=0, p_set=0.2)

n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots, oltc_control=True) # True # False

n.transformers_t.p0

print(n.buses_t.v_mag_pu)


for ind, row in n.transformers.iterrows():
    opt_tap = row['tap_position']  # initial value for tap position
    # extracting controlled nodes names
    nodes = [x.strip() for x in n.transformers['ctrl_nodes'][0].split(',')]
    # if no node is chosen take node of secondary of trafo as control node
    ctrl_nodes = np.where(
            len(n.transformers['ctrl_nodes'][0]) == 0, [row['bus1']], nodes)
    # find voltages of controlled nodes
    # v_pu_ctrl_nodes = n.buses_t.v_mag_pu.loc[snapshot, ctrl_nodes]
    # find taps and tap_steps
    if row['type'] != '':
        taps = (np.arange(n.transformer_types.loc[row['type'], 'tap_min'],
                         n.transformer_types.loc[row['type'], 'tap_max']+1))

        tap_side = n.transformer_types.loc[row['type'], 'tap_side']
        tap_step = n.transformer_types.loc[row['type'], 'tap_step']

    else:
        taps = np.arange(row['tap_min'], row['tap_max']+1)
        tap_side = row['tap_side']
        tap_step = row['tap_step']

    tap_side_cons = np.where(tap_side == 0, 1, -1)
    # Single node oltc control part:
    if len(ctrl_nodes) == 1:

        deadband_range = row['v_set'] + np.array([row['deadband']/100, -row[
                'deadband']/100])
## Example of transformer with non-trivial phase shift and tap ratio
#
#This example is a copy of pandapower's minimal example.

# def apply_oltc(n, snapshot):
#     n=network
#     for ind, row in n.transformers.iterrows():
#         tap_side_cons = np.where(row['tap_side'] == 0, 1, -1)
#         opt_tap = row['tap_position']  # initial value for tap position
#         # extracting controlled nodes names
#         nodes = [x.strip() for x in n.transformers['ctrl_nodes'][0].split(',')]
#         # if no node is chosen take node of secondary of trafo as control node
#         ctrl_nodes = np.where(
#                 len(n.transformers['ctrl_nodes'][0]) == 0, [row['bus1']], nodes)
#         # find voltages of controlled nodes
#         v_pu_ctrl_nodes = n.buses_t.v_mag_pu.loc["now", ctrl_nodes]
#         # find taps and tap_steps
#         if row['type'] != '':
#             taps = (np.arange(n.transformer_types.loc[row['type'], 'tap_min'],
#                              n.transformer_types.loc[row['type'], 'tap_max']+1))

#             tap_step = n.transformer_types.loc[row['type'], 'tap_step']

#         else:
#             taps = np.arange(row['tap_min'], row['tap_max']+1)
#             tap_step = row['tap_step']
#         # Multiple node oltc control part:
#         if len(ctrl_nodes) > 1:
#             # find the mean and max voltages of the controlled nodes
#             meas_max = v_pu_ctrl_nodes.values.max()
#             meas_min = v_pu_ctrl_nodes.values.min()
#             # check if meas_max and meas_min are withing the range
#             if (meas_min > row['v_min'] and meas_max < row['v_max']):
#                 print('they are within the range')
#             else:
#                 max_voltage = meas_max-taps*tap_step*row['v_set']/100
#                 min_voltage = meas_min-taps*tap_step*row['v_set']/100
#                 opt_ind = np.where(((min_voltage > row['v_min']) & (
#                         max_voltage < row['v_max'])))[0]

#                 if len(opt_ind) != 0:
#                     opt_tap = taps[opt_ind[0]]
#                 else:
#                     opt_ind = np.where(min_voltage > row['v_min'])[0]
#                     if len(opt_ind) != 0:
#                         opt_tap = taps[len(opt_ind)-1]

#                     else:
#                         opt_tap = taps[0]
#     return opt_tap


# optimum = apply_oltc(network, "2020-01-01 00:00")