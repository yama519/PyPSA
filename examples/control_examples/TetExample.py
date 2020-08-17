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
n.add("Transformer", "MV-LV trafo", type="0.63 MVA 10/0.4 kV", bus0="MV bus", bus1="LV1 bus", tap_side=1)
n.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
n.add("Generator", "External Grid", bus="MV bus", control="Slack")
n.add("Load", "LV load", bus="LV1 bus", p_set=1, s_nom=1.5, power_factor_min=0.9, control_strategy = '')
n.add("Load", "test load", bus="LV1 bus", p_set=0, s_nom=1.3)
n.add("Load", "test load2", bus="LV2 bus", p_set=1, s_nom=1.2, power_factor=0.1, control_strategy = '')  
n.add("Load", "test load3", bus="LV1 bus", p_set=1, s_nom=1.3, power_factor=0.3, control_strategy = '')
n.add("Generator", "PV", bus="LV2 bus", control="PQ", p_set =1, q_set = 0, s_nom=1.5, power_factor=0.9, control_strategy = '')
n.add("Load", "test load4", bus="LV1 bus", p_set=1, q_set=1)
n.add("Store", "store2", bus="LV2 bus", p_set=0,q_set=0)
n.add("StorageUnit", "store2", bus="LV2 bus", p_set=0, control_strategy = '', power_factor=0.90)
n.add("StorageUnit", "store1", bus="LV2 bus", q_set=0, p_set=-L, control_strategy = '')

n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots)  # False  # True

n.transformers.oltc=True
n.lpf(n.snapshots)
n.pf(use_seed=True, snapshots=n.snapshots, oltc_control=True)  



