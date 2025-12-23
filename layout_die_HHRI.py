#pip install gdsfactory==7.22.3

import gdsfactory as gf

from library_sipho_unified import *
from library_sipho_params import *
from library_electrode_unified import *
from library_electrode_params import *
import pandas as pd

combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_taper": True, "DC_MT1": False, "RF_out": True,
                   "PS_length": 850,
                   "trans_length": 200,
                    "trans_length_to_term": 200,
                    "PS_trans_length": 100,
                   "htr_bridge_place": True,
                   "htr_bridge_width": 4,
                   "htr_bridge_pitch": 70,
                   "htr_bridge_reps": 3,
                   }
# temp fixes and offsets
combined_params["wsclDOEcell_placeGC"] = 1
# myXoff = 350
myXoff = 70
myYoff = 1230

# here the cell building starts
c = gf.Component("Mirach_DOE_1_2025_12_19_v7")

# read and place scribe lanes
scribeLane = gf.import_gds("DOEdieFrame.gds", with_metadata=True)
c << scribeLane

# read DOE file (firs row has parameter names, other rows have values)
myDOETable = pd.read_csv("DOE1_pars.csv")

# assign parameters and create cells
for j in range(len(myDOETable)):
#for j in range(1):
    for i, par in enumerate(myDOETable.head()):
        # skip empty fields
        if myDOETable.iat[j,i]!="":
            combined_params[par] = myDOETable.iat[j,i]
        #print(par)
    # place DOE elements
    mycell=wsclDOEcell(combined_params)
    _ = c << mycell
    # read placement coordinates and move
    X, Y = combined_params["X"], combined_params["Y"]
    _.move((X+myXoff, Y+myYoff))

# c = merge_clad(c, 0)
# c.name = "Mirach_DOE_1_2025_12_14_v2"
c.show()



