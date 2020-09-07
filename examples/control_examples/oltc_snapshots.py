# -*- coding: utf-8 -*-

from __future__ import print_function, division
import pandas as pd
import pypsa
import numpy as np
from pypsa.descriptors import get_switchable_as_iter

snapshot = pd.date_range(start="2020-01-01 00:00", end="2020-01-01 12:00",  periods=5)
L = pd.Series([-0.1]*1 + [0.2]*1 + [0.3]*1 + [0.4]*1 + [0.5]*1, snapshot)

n = pypsa.Network()
n.set_snapshots(snapshot)
n.add("Bus", "MV bus", v_nom=20, v_mag_pu_set=1.02)
n.add("Bus", "LV1 bus", v_nom=.4)
n.add("Bus", "LV2 bus", v_nom=.4)
#n.add("Bus", "LV3 bus", v_nom=.4)
n.add("Transformer", "MV-LV trafo", s_nom=0.63, x=0.0385161, r=0.010794, g=0.00187302,
      bus0="MV bus", bus1="LV1 bus", oltc=True, tap_side=0)

n.add("Line", "LV cable", type="NAYY 4x50 SE", bus0="LV1 bus", bus1="LV2 bus", length=0.1)
n.add("Generator", "External Grid", bus="MV bus", control="Slack")
n.add("Generator", "test load2", bus="LV2 bus", p_set=L, s_nom=1.2, power_factor=0.1, control_strategy = '')  
n.add("Generator", "test load3", bus="LV1 bus", p_set=L, s_nom=1.3, power_factor=0.3, control_strategy = '')
n.add("Generator", "PV", bus="LV2 bus", control="PQ", p_set =L, control_strategy = '')


def run_pf(oltc_control=False):
    n.lpf(n.snapshots)
    n.pf(use_seed=True, snapshots=n.snapshots, oltc_control=oltc_control)

# run_pf()
# v_mag_no = n.buses_t.v_mag_pu
# n.transformers_t.opt_tap_position
# df = n.transformers.tap_position

print(n.buses_t.v_mag_pu)

run_pf(oltc_control=True)
v_mag_oltc = n.buses_t.v_mag_pu
trafodf = n.transformers
df = n.transformers
