# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pypsa
import numpy as np
import pandas as pd
n = pypsa.Network()
Generators_injection = pd.DataFrame(columns=[])
Bus_v_mag_pu = pd.DataFrame(columns=[])
n.add("Bus","MV bus",v_nom=20,v_mag_pu_set=1.02)
n.add("Bus","LV1 bus",v_nom=.4)
n.add("Bus","LV2 bus",v_nom=.4)


n.add("Transformer","MV-LV trafo",type="0.63 MVA 20/0.4 kV",bus0="MV bus",bus1="LV1 bus")

n.add("Line","LV cable",type="NAYY 4x50 SE",bus0="LV1 bus",bus1="LV2 bus",length=0.1)

n.add("Generator","External Grid",bus="MV bus",control="Slack")
n.add("Generator","My gen1",bus="LV1 bus",control="PQ", p_set=0.1)
n.add("Generator","My gen2",bus="LV2 bus",control="PQ", p_set=0.2)

n.add("Load","LV load1",bus="LV1 bus",p_set=0)
n.add("Load","LV load2",bus="LV2 bus",p_set=0)

def run_pf(oltc_control=False):
    
    n.lpf()
    n.pf(use_seed=True, oltc_control=oltc_control)
    if not oltc_control:
        Generators_injection['no_control'] = n.buses_t.p.T['now'].loc["LV1 bus":"LV2 bus"]
        Bus_v_mag_pu['no_control'] = n.buses_t.v_mag_pu.T['now'].loc["LV1 bus":"LV2 bus"]
    if oltc_control:
        Bus_v_mag_pu['with_control'] = n.buses_t.v_mag_pu.T['now'].loc["LV1 bus":"LV2 bus"]



run_pf()
v_mag_no = n.buses_t.v_mag_pu

n.transformers.oltc = True
n.transformers.deadband = 2
n.transformers.tap_step = 2.5
n.transformers.ctrl_nodes = "LV1 bus"
run_pf(oltc_control=True)
v_mag_with = n.buses_t.v_mag_pu



def plot(figure):
    plt.figure(figure)
    plt.plot(Generators_injection['no_control'], Bus_v_mag_pu['no_control'],
              linestyle='--', label="no_controller applied", color="r")
    plt.scatter(Generators_injection['no_control'], Bus_v_mag_pu['no_control'], color="r")
    
    
    plt.plot(Generators_injection['no_control'], Bus_v_mag_pu['with_control'],
              linestyle='--', label="oltc applied", color="g")
    plt.scatter(Generators_injection['no_control'], Bus_v_mag_pu['with_control'], color="g")
    # plt.yticks([1.00413, 1.10166, 1.02885, 1.07849])
    # plt.yticks([0.980577, 1.10166, 1.02885, 1.05649])
    plt.yticks(Bus_v_mag_pu['no_control'].append(Bus_v_mag_pu['with_control']))
    plt.xticks(Generators_injection['no_control'], rotation=70)
    
    plt.axhline(y=1.02, color='y', linestyle='--', label='upper limit deadband: 1.02',
                linewidth=3, alpha=0.5)
    plt.axhline(y=0.98, color='r', linestyle='--', label='lower limit deadband: 0.98',
                linewidth=3, alpha=0.5)
    
    plt.xlabel('Active_power_injection (MW)')
    plt.ylabel('V_mag_pu (per unit)')
    
    plt.title("Voltage rise due to increase in active power injection")
    
    plt.legend()
    plt.show()

plot(1)


n.transformers.tap_position=0
run_pf()
n.transformers.oltc = True
n.transformers.deadband = 2
n.transformers.tap_step = 2.5
n.transformers.ctrl_nodes = "LV1 bus, LV2 bus"
run_pf(oltc_control=True)

plot(2)