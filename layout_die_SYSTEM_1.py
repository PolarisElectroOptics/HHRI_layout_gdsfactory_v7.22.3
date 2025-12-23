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
combined_params = {**combined_params,

                "PS_length": 600,
                "w_slot": 0.16,
                "w_slotWG": 0.18,
                "w_OXOP" : 5,
                "s2s_type": "MMI",

                "pad_outer_gnd_width": 102.5,
                "S2S_center_gnd_width": 50,
                "S2S_inner_gap_width": 15,
                "S2S_sig_width": 37,
                "S2S_outer_gap_width": 4.5,
                "S2S_outer_gnd_width": 231- 40,
                "S2S_length": 50,

                "MT1_from_PS": False,  # places grounds from PS_slotWG rather than central one in GSGSG_MT2
                "PS_center_gnd_width": 50,
                "PS_inner_gap_width": 15,
                "PS_sig_width": 37,
                "PS_outer_gap_width": 70,  # used for GSGSG only
                "PS_outer_gnd_width": 165.5-40, #used for GSGSG only

                "PS_trans_length": 100
                }


c = gf.Component("Mirach_system_die_1_2025_12_16_v6")

die_template = gf.import_gds("frame_bump251028a_PEO_update_PS600_12_8.gds", with_metadata=True)
#die_template = gf.add_ports.add_ports_from_markers_inside(die_template, pin_layer=(1011,0), port_layer=MT2)
c << die_template


array, array_refs = PEO_place_array(MZM_GSGSG_new, combined_params, None, n_rows=1, n_cols=4, pitch_x=625, pitch_y=3000, rotation=90)
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




