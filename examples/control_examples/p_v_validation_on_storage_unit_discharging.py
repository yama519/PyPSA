from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
'''
This example is for validation of active power curtailment or P(V) control
on StorageUnit components while discharging, the plot shows the controller output
validation against its droop characteristic.
'''
snapshots = pd.date_range(start="2020-01-01 00:00", end="2020-01-01 02:00",
                          periods=15)

# power generation or injection
L = pd.Series([0.00001]*1+[0.00002]*1+[0.00003]*1+[0.00005]*1+[0.00007]*2 +
              [0.00008]*1+[0.0001]*2+[0.00021]*2+[0.005]*2+[0.007]*2,
              snapshots)
# load
L1 = pd.Series([0.00582]*1+[0.00553]*1+[0.0053]*1+[0.0045]*1+[0.004]*2+[0.0035]*1 +
               [0.00499]*1+[0.0047]*1+[0.0043]*1+[0.002]*1+[0.0008]*1 +
               [0.00001]*1+[0.]*1+[0.0]*1, snapshots)

# building empty dataframes
Results_v = pd.DataFrame(columns=[])
droop = pd.DataFrame(columns=[])

# defining network
n = pypsa.Network()
n.set_snapshots(snapshots)
n_buses = 30

for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4, v_mag_pu_set = 1.03)
for i in range(n_buses):
    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ", p_set=0, v_pu_cr=0.95, v_max_curtail=1.02)
    n.add("StorageUnit", "My store {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ", p_set=L, v_pu_cr=0.95, v_max_curtail=1.02)
    n.add("Load", "My load {}".format(i), bus="My bus {}".format(
        (i) % n_buses), p_set=L1)
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)
# n.generators.type_of_control_strategy = 'p_v'
def power_flow():
    n.lpf(n.snapshots)
    n.pf(use_seed=True, snapshots=n.snapshots, x_tol_outer=1e-6, inverter_control=True)


# setting control strategy type
n.storage_units.control_strategy = 'p_v'
power_flow()


# saving the necessary data for plotting controller behavior
Results_v = n.buses_t.v_mag_pu.loc[:, 'My bus 1':'My bus 29']

Results_p = n.storage_units_t.p.loc[:, 'My store 1':'My store 29'] / \
        n.storage_units_t.p_set.loc[:, 'My store 1':'My store 29'] * 100

# controller droop characteristic method simplified
def p_v(v_pu_bus):
    v_pu_cr = n.storage_units.loc['My store 26', 'v_pu_cr']
    v_max_curtail = n.storage_units.loc['My store 26', 'v_max_curtail']

    pperpmax = np.select([(v_pu_bus < v_pu_cr), (v_pu_bus > v_max_curtail), (
        v_pu_bus >= v_pu_cr and v_pu_bus <= v_max_curtail)], [100, 0, (
            100-(100/(v_max_curtail-v_pu_cr))*(v_pu_bus-v_pu_cr))])
    return pperpmax

Results_droop = pd.DataFrame(np.arange(0.92, 1.04, 0.01), columns=['v_mag'])
# calculation of droop characteristic (output)
for index, row in Results_droop.iterrows():
    Results_droop.loc[index, "p/pmaxdroop"] = p_v(Results_droop.loc[index, 'v_mag'])
# ''' Plotting '''
# plotting controller behavior over droop as dots
plt.plot(Results_droop['v_mag'], Results_droop["p/pmaxdroop"],
         label="P(V) droop characteristic", color='r')
plt.scatter(Results_v, Results_p, color="g",
            label="P(V) Controller characteristic\n critical_voltage=0.95")

plt.legend(loc="best", bbox_to_anchor=(0.4, 0.4))
plt.title("P(V) Controller Behavior in a 30 node example, \n snapshots = 15 ")
plt.xlabel('voltage_pu')
plt.ylabel('Active Power after Curtailment %')
plt.show()
