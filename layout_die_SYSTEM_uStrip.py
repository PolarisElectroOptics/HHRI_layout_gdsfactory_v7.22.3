#pip install gdsfactory==7.22.3

import gdsfactory as gf

from library_sipho_unified import *
from library_sipho_params import *
from library_electrode_unified import *
from library_electrode_params import *


# c = gf.Component("Mirach_PS_connnected_sample_20251124_v2")
# _ = c << PS_connected_from_params(combined_params)
# c.show()
#
# c = gf.Component("Mirach_MZM_GSG_sample_20251124")
# combined_params = {**balun_electrode_params, **balun_sipho_params}
# _ = c << MZM_GSG_balun(combined_params)
# _.rotate(-90)
# c.show()

combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_taper": True, "DC_MT1": True, "RF_out": False}
combined_params["PS_length"] = 1200
combined_params["S2S_center_gnd_width"] = 2
combined_params["PS_center_gnd_width"] = 2
combined_params["S2S_inner_gap_width"] = 20
combined_params["S2S_sig_width"] = 20
combined_params["S2S_outer_gap_width"] = 40
combined_params["S2S_outer_gnd_width"] = 100
combined_params["S2S_length"] = 10

 # parameters from design document
# M2
combined_params["g2_Pad"] = 25
combined_params["w_Pad"] = 100
combined_params["g1_Pad"] = 25
combined_params["w_c_Pad"] = 50
combined_params["l_Taper"] = 200
combined_params["g2_PS"] = 59.5
combined_params["w_PS"] = 63
combined_params["g1_PS"] = 50
combined_params["l_S2S"] = 100 
combined_params["Pitch"] = 100

combined_params["h_pad"] = 55
combined_params["g_padW"] = 100

# VIA1
combined_params["w_V1"] = 5
combined_params["g2_V1"] = 100.6
combined_params["g1_V1"] = 50.6

# M1
combined_params["w2_M1"] = 10
combined_params["g2_M1"] = 50
combined_params["w1_M1"] = 20
combined_params["g1_M1"] = 24
combined_params["w_c_M1"] = 2



c = gf.Component("Mirach_system_die_uStrip_v1")

match combined_params['PS_length']:
    case 600:
        myFrame="frame_bump251028a_PEO_update_PS600_12_8.gds"
    case 800:
        myFrame="frame_bump251028a_PEO_update_PS800_12_10.gds"
    case 1200:
        myFrame="frame_bump251028a_PEO_update_PS1200_12_10.gds"

die_template = gf.import_gds(myFrame, with_metadata=True)
#die_template = gf.add_ports.add_ports_from_markers_inside(die_template, pin_layer=(1011,0), port_layer=MT2)
c << die_template


array, array_refs = PEO_place_array(MZM_GSGSG_uStrip, combined_params, None, n_rows=1, n_cols=4, pitch_x=625, pitch_y=3000, rotation=90)
_ = c << array
_.move((630+30, 60 ))

_ = c << array
_.move((630+30+3125, 60 ))


c.show()

#
# c = gf.Component("Mirach_sample_MZM_GSGSG_2025_12_09")
# _ = c << MZM_GSGSG_new(combined_params)
# c.show()


# c = gf.Component("Mirach_system_MZM_only_sample_MZM_GSGSG_2025_12_08_v6")
# combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_taper": True, "DC_MT1": True, "RF_out": False}
# combined_params["PS_length"] = 600
#
# #die_template = gf.import_gds("frame_bump251028a_PEO_update_12_8.gds", with_metadata=True)
# #die_template = gf.add_ports.add_ports_from_markers_inside(die_template, pin_layer=(1011,0), port_layer=MT2)
# #c << die_template
#
#
# array, array_refs = PEO_place_array(MZM_GSGSG_new, combined_params, None, n_rows=1, n_cols=4, pitch_x=625, pitch_y=3000, rotation=90)
# _ = c << array
# _.move((630+30, 60 ))
#
# c.show()




