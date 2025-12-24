#pip install gdsfactory==7.22.3

import gdsfactory as gf

from library_sipho_unified import *
from library_sipho_params import *
from library_electrode_unified import *
from library_electrode_params import *
import pandas as pd

combined_params = {**differential_electrode_params, **balun_sipho_params, "gsgsg_variant": "DC", "s2s_type": "adiabatic",  #MMI, adiabatic, power
                    "PS_length": 50,
                    #"PS_sipho_length": 900,
                    "w_slot": 0.20,
                    "w_slotWG": 0.20,
                    "S2S_ADIA_W": 0.4,
                    "S2S_ADIA_W0": 0.18,
                    "S2S_ADIA_R": 0.2, #default to w_slotWG
                    "S2S_ADIA_S": 0.17, #default to w_slot
                    "S2S_ADIA_G": 0.17,
                    "S2S_ADIA_L1": 4,
                    "S2S_ADIA_L2": 4,
                    "S2S_ADIA_L3": 42,
                    "S2S_ADIA_B1": 2.25,
                    "S2S_ADIA_B2": 0.87,
                    "S2S_ADIA_B3": 5,
                    "S2S_ADIA_C1": 0.525,
                    "S2S_ADIA_C2": 0.13,

                    "pad_center_gnd_width": 80,
                    "pad_inner_gap_width": 20,
                    "pad_sig_width": 80,
                    # "pad_outer_gap_width": 20,
                    # "pad_outer_gnd_width": 80,  # 200-7.5,
                    "pad_length": 150,

                    "S2S_center_gnd_width": 50,
                    "S2S_inner_gap_width": 16,
                    "S2S_sig_width": 40,
                    "S2S_outer_gap_width": 5,
                    "S2S_outer_gnd_width": 60,
                    "S2S_length": 50,

                    "PS_center_gnd_width": 50,
                    "PS_inner_gap_width": 16,
                    "PS_sig_width": 40,
                    "PS_outer_gap_width": 15, #used for GSGSG only
                    "PS_outer_gnd_width": 50,#-40, #used for GSGSG only
                    "trans_length": 250,

                    "pad_t_length": 60,
            }

# here the cell building starts
c = gf.Component("TOP")



myXoff = 150
myYoff = 600



# read and place scribe lanes
scribeLane = gf.import_gds("HHRIdieFrame.gds", with_metadata=True)
c << scribeLane

# read DOE file (firs row has parameter names, other rows have values)
myDOETable = pd.read_csv("LSMZM_pars.csv")



# assign parameters and create cells
for j in range(len(myDOETable)):
#for j in range(1):
    for i, par in enumerate(myDOETable.head()):
        # skip empty fields
        if myDOETable.iat[j,i]!="":
            combined_params[par] = myDOETable.iat[j,i]
        #print(par)
    # place DOE elements
    mycell=MZM_SilTerra(combined_params)
    _ = c << mycell
    _.rotate(-90)
    # read placement coordinates and move
    X, Y = combined_params["X"], combined_params["Y"]
    _.move((X+myXoff, Y+myYoff))

# c = merge_clad(c, 0)
# c.name = "Mirach_DOE_1_2025_12_14_v2"


c.show()



