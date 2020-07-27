"""
This example shows the effect of application of each controller and mix of them
on StorageUnit component while charging. The effect of each controller and mix of them are
compared in a graph against no control case. The discharging effect can be
also seen by changing the sign of p_set to positive.
"""
# importing important modules
from __future__ import print_function, division
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

power_inj = [0, 0.03, 0.05, 0.07]
#power_inj = [0, 0.01, 0.02, 0.06]
Load = [0.1, 0.14, 0.18, 0.23]

# defining network
n = pypsa.Network()

n_buses = 4

for i in range(n_buses):
    n.add("Bus", "My bus {}".format(i), v_nom=.4, v_mag_pu_set=1.02)

for i in range(n_buses):
    n.add("Generator", "My Gen {}".format(i), bus="My bus {}".format(
        (i) % n_buses), control="PQ")

    n.add("StorageUnit", "My storage_unit {}".format(i), bus="My bus 1".format(
        (i) % n_buses), p_set=-power_inj[i], v1=0.89, v2=0.99, v3=0.96, v4=1.01,
        s_nom=0.075, set_p1=50, set_p2=100, power_factor=0.95,
        power_factor_min=0.98, p_ref=0.075)

for i in range(n_buses-1):
    n.add("Line", "My line {}".format(i), bus0="My bus {}".format(i),
          bus1="My bus {}".format((i+1) % n_buses), x=0.1, r=0.01)


def run_pf(inverter_control=False):
    n.lpf()
    n.pf(use_seed=True, inverter_control=inverter_control)

# run pf without controller and save the results
run_pf()

Storage_injection = pd.DataFrame(columns=[])
Bus_q_values = pd.DataFrame(columns=[], index=n.buses.index)
storage_q_value = pd.DataFrame(columns=[], index=n.storage_units.index)
Bus_v_mag_pu = pd.DataFrame(columns=[])
Storage_injection['no_control'] = n.storage_units_t.p.T['now']
Bus_v_mag_pu['no_control'] = n.buses_t.v_mag_pu.T['now']
Bus_q_values['no_control'] = n.buses_t.q.T['now'].values
storage_q_value['no_control'] = n.storage_units_t.q.T['now'].values


# now apply reactive power as a function of voltage Q(U) or q_v controller,
# parameters (v1,v2,v3,v4,s_nom,damper) are already set in (n.add('StorageUnit', ...))
n.storage_units.control_strategy = 'q_v'
run_pf(inverter_control=True)
Storage_injection['q_v_control'] = n.storage_units_t.p.T['now']
Bus_v_mag_pu['q_v_control'] = n.buses_t.v_mag_pu.T['now']
Bus_q_values['q_v_control'] = n.buses_t.q.T['now'].values
storage_q_value['q_v_control'] = n.storage_units_t.q.T['now'].values


### now apply fixed power factor controller (fixed_cosphi), parameters
### (power_factor, damper) are already set in (n.add(Generator...))
n.storage_units.control_strategy = 'fixed_cosphi'

# run pf and save the results
run_pf(inverter_control=True)
Storage_injection['fixed_pf_control'] = n.storage_units_t.p.T['now']
Bus_v_mag_pu['fixed_pf_control'] = n.buses_t.v_mag_pu.T['now']
Bus_q_values['fixed_pf_control'] = n.buses_t.q.T['now'].values
storage_q_value['fixed_pf_control'] = n.storage_units_t.q.T['now'].values


### now apply power factor as a function of real power (cosphi_p), parameters
### (set_p1,set_p2,s_nom,damper,power_factor_min) are already set in (n.add('StorageUnit'...))
##
n.storage_units.control_strategy = 'cosphi_p'

# run pf and save the results
run_pf(inverter_control=True)
Storage_injection['cosphi_p_control'] = n.storage_units_t.p.T['now']
Bus_v_mag_pu['cosphi_p_control'] = n.buses_t.v_mag_pu.T['now']
Bus_q_values['cosphi_p_control'] = n.buses_t.q.T['now'].values
storage_q_value['cosphi_p_control'] = n.storage_units_t.q.T['now'].values
## now apply mix of controllers



## fixed_cosphi controller
n.storage_units.loc['My storage_unit 1', 'control_strategy'] = 'q_v'
## fixed_cosphi controller
n.storage_units.loc['My storage_unit 2', 'control_strategy'] = 'fixed_cosphi'

# cosphi_p controller
n.storage_units.loc['My storage_unit 3', 'control_strategy'] = 'cosphi_p'

run_pf(inverter_control=True)
#
Storage_injection['mix_controllers'] = n.storage_units_t.p.T['now']
Bus_v_mag_pu['mix_controllers'] = n.buses_t.v_mag_pu.T['now']
Bus_q_values['mix_controllers'] = n.buses_t.q.T['now'].values
storage_q_value['cosphi_p_control'] = n.storage_units_t.q.T['now'].values

# plotting
plt.plot(-Storage_injection['no_control'], Bus_v_mag_pu['no_control'],
         linestyle='--', label="no_control")
plt.plot(abs(Storage_injection['q_v_control']), Bus_v_mag_pu['q_v_control'],
         label="q_v")

plt.plot(abs(Storage_injection['fixed_pf_control']), Bus_v_mag_pu['fixed_pf_control'],
         label="fixed_cosphi")
plt.plot(abs(Storage_injection['cosphi_p_control']), Bus_v_mag_pu['cosphi_p_control'],
         label="cosphi_p")

plt.plot(abs(Storage_injection['mix_controllers']), Bus_v_mag_pu['mix_controllers'],
         label="mix")
plt.axhline(y=1.02, color='y', linestyle='--', label='max_v_mag_pu_limit',
            linewidth=3, alpha=0.5)
plt.axhline(y=0.9, color='r', linestyle='--', label='min_v_mag_pu_limit',
            linewidth=3, alpha=0.5)
plt.xlabel('Active_power_injection (MW)')
plt.ylabel('V_mag_pu (per unit)')
plt.title("Application of controllers and mix of them on Storage_U component")
plt.legend()
plt.show()
