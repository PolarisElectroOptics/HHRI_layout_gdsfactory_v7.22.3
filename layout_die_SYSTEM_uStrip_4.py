#pip install gdsfactory==7.22.3
# layout file for system die #4

import gdsfactory as gf

from library_sipho_unified import *
from library_sipho_params import *
from library_electrode_unified import *
from library_electrode_params import *

combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_taper": True, "DC_MT1": True, "RF_out": False}
combined_params["PS_length"] = 600
combined_params["S2S_center_gnd_width"] = 2
combined_params["PS_center_gnd_width"] = 2
combined_params["S2S_inner_gap_width"] = 20
combined_params["S2S_sig_width"] = 20
combined_params["S2S_outer_gap_width"] = 40
combined_params["S2S_outer_gnd_width"] = 100
combined_params["S2S_length"] = 10
combined_params["w_slot"] = 0.16
combined_params["w_slotWG"] = 0.18
combined_params["s2s_type"] = "adiabatic"
combined_params["w_OXOP"] = 5
combined_params["w_IND"] = 12.2+1.3
combined_params["w_NCONT"] = 12.2+1.3
#combined_params["gap_si_contact"] = 0.015-0.03+0.5+0.055
combined_params["w_si_contact"] = 13.5+1.5


 # parameters from design document
combined_params["g1_PS"] = 50
combined_params["g2_PS"] = 55
combined_params["w_M2_PS"] = 70
combined_params["w_M1_PS"] = 25
combined_params["g2_M1_PS"] = 100
combined_params["l_PS"] = 600
combined_params["g1_S2S"] = 50
combined_params["g2_S2S"] = 45
combined_params["w_M2_S2S"] = 80
combined_params["w_M1_S2S"] = 25
combined_params["l_S2S"] = 100
combined_params["g1_Taper2"] = 50
combined_params["w_M1_Taper2"] = 25
combined_params["g1_Taper1"] = 90
combined_params["w_M1_Taper1"] = 15
combined_params["l_Taper"] = 200
combined_params["g1_Pad"] = 20
combined_params["g2_Pad"] = 20
combined_params["w_M2_Pad"] = 110
combined_params["w_C_Pad"] = 50

combined_params["h_pad"] = 55
combined_params["g_padW"] = 100

combined_params["gap_NCONT_WG"] = 2.0


combined_params["S2S_ADIA_R"] = 0.18 
combined_params["S2S_ADIA_S"] = 0.16 
combined_params["S2S_ADIA_G"] = 0.16
combined_params["S2S_ADIA_B3"] = 2

c = gf.Component("Mirach_system_die_uStrip_4_v5")


myFrame="frame_bump251028a_PEO_update_PS1200_4Die6.gds"


die_template = gf.import_gds(myFrame, with_metadata=True)
#die_template = gf.add_ports.add_ports_from_markers_inside(die_template, pin_layer=(1011,0), port_layer=MT2)
c << die_template


array, array_refs = PEO_place_array(MZM_GSGSG_uStrip, combined_params, None, n_rows=1, n_cols=4, pitch_x=625, pitch_y=3000, rotation=90)
_ = c << array
_.move((630+30, 60 ))

_ = c << array
_.move((630+30+3125, 60 ))


c.show()




