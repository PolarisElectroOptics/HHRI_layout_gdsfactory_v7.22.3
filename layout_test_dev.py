#pip install gdsfactory==7.22.3

import gdsfactory as gf
#from functools import partial
#import math
#from layer_info import *
#from PDK_AMF import *
#from library_EC import *

from library_sipho_unified import *
from library_sipho_params import *
from library_electrode_unified import *
from library_electrode_params import *

import shapely
from shapely.geometry.polygon import Polygon


c = gf.Component("Test_SilTerra")
combined_params = {**differential_electrode_params, **SilTerra_sipho_params, "gsgsg_variant": "DC", "s2s_type": "adiabatic",  #MMI, adiabatic, power

                    "w_slot": 0.20,
                    "w_slotWG": 0.20,
                    "buffer_RIB_SLAB_overlay": 0.2,
                    "buffer_ETCH_HM_overlay": 0.1,
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

                    "pad_length": 100,
                    "trans_length": 150,                
                    "S2S_length": 50,
                    "PS_length": 50,                   

                    "PS_center_gnd_width": 63,  #coupling_length +(bend_r-7.5)*2
                    "PS_inner_gap_width": 15,
                    "PS_sig_width": 40,
                    
            }

_ = c << MRM_SilTerra(combined_params)
_.movex(500).movey(250)


combined_params = {**differential_electrode_params, **SilTerra_sipho_params, "gsgsg_variant": "DC", "s2s_type": "adiabatic",  #MMI, adiabatic, power

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

                    "pad_length": 100,
                    "trans_length": 150,                
                    "S2S_length": 50,
                    "PS_length": 1000,                   

                    "PS_center_gnd_width": 63,  
                    "PS_inner_gap_width": 15,
                    "PS_sig_width": 40,
                    
                    "S2S_center_gnd_width": 63,  
                    "S2S_inner_gap_width": 15,
                    "S2s_sig_width": 40,                  
                    
                    
            }

_ = c << MZM_SilTerra(combined_params)
_.rotate(90)

# _ = c << s2s_adiabatic(combined_params)

# _ = c << GSG_MRM_SilTerra(combined_params)

c.show()






# combined_params={**wscl2_params,**electrode_wscl2_params, **differential_electrode_params, **balun_sipho_params, "wsclDOEcell_placeGC":False, "PS_length": 1000}
#
# c = gf.Component("s2s_withpowertaper_rail")
# combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True}
# combined_params["PS_length"] = 1000
# combined_params["RF_out"] = 0
# # combined_params["extra_slab"]=1
#
# #_ = c << s2s_from_params(params = combined_params, non_taper = 0.3)
#
#
# _ = c << s2s_from_params_powertaper(params = combined_params, taper_pwr = 0.1)
# c.show()

#Mirach layout test
#combined_params = {**DOE6_params, **electrode_DOE6_params}
# #
# combined_params={**wscl2_params,**electrode_wscl2_params, **differential_electrode_params, **balun_sipho_params, "wsclDOEcell_placeGC":False, "PS_length": 1000, "RF_out": True}
#
# c = gf.Component("wsclDOEcell")
# _ = c << wsclDOEcell(combined_params)
# _.pprint_ports()
# c.show()
# #
# array = wsclDOEarray(combined_params, None, n_cols = 3, pitch_x=2200, name="wsclDOEarray", rotation=0, GC_array_movex=3500, GC_array_movey=-1700)
# array.name = "wsclDOEarray"
# array.add_label(text="ORIGIN", position=(0, 0), magnification=40, layer=MT2) #dummy label somehow required to get subcell magnification to work too..
# array.show()
#


# combined_params = {**balun_electrode_params, **balun_sipho_params}
# combined_params = {**combined_params, "termination": 84}#, "gnds_shorted": True}
# b = gf.Component("TC_DOE_new")
# combined_params["electrical"] = True
# tc=b<<TC_DOE_new(combined_params,'DOE6_CSV_Params/TC_DOE_pars.csv',3)
#
# c= merge_clad(b, 0)
# c.show()

#
# combined_params = {**differential_electrode_params, **balun_sipho_params}
# combined_params = {**combined_params, "termination": 84}#, "gnds_shorted": True}

# c = gf.Component("TC_DOE")
#tc = c << TC_DOEcell_from_csv_params(combined_params, "DOE6_CSV_Params/TC_DOE_pars.csv", 1)





# c = gf.Component("TC_DOE_new")
# combined_params["electrical"] = False
# tc=c<<TC_DOE_new(combined_params,'DOE6_CSV_Params/TC_DOE_pars.csv',3)
# c.show()


# c = gf.Component("TC_DOE_w_electrical")
# TC_params = combined_params
# TC_params["electrical"] = True
# TC_params["min_gap_OXOP_MT"] =2# -2
# tc=c<<TC_DOE_new(TC_params,'DOE6_CSV_Params/TC_DOE_pars.csv',3)
# c.show()
#


# c = gf.Component("Mirach_PS_connnected_sample_20251124_v2")
# _ = c << PS_connected_from_params(combined_params)
# c.show()
#
# c = gf.Component("Mirach_MZM_GSG_sample_20251124")
# combined_params = {**balun_electrode_params, **balun_sipho_params}
# _ = c << MZM_GSG_balun(combined_params)
# _.rotate(-90)
# c.show()


# c = gf.Component("test_vias")
# c << PEO_custom_via_from_line()
#
# c.show()

# c = gf.Component("fill test")
# @gf.cell
# def a():
#     c = gf.Component()
#     x_points = [0, 0, 20, 20]
#     y_points = [0, 7, 10, 0]
#
#     #c.add_polygon([x_points, y_points], layer=MT1)
#
#     P_shapely = Polygon(list(zip(x_points, y_points)))
#     P_shapely_shrink = P_shapely.buffer(-3)
#     #c.add_polygon(P_shapely_shrink, layer=MT1)
#     P_shapely_final = shapely.difference(P_shapely, P_shapely_shrink)
#     a = P_shapely_final.bounds
#     P_shapely_subtract = Polygon([(a[2]-3, a[1]+3), (a[2]-3, a[3]-3), (a[2], a[3]-3), (a[2], a[1]+3)] )
#     P_shapely_final = shapely.difference(P_shapely_final, P_shapely_subtract)
#     c.add_polygon(P_shapely_subtract, layer=MT2)
#     c.add_polygon(P_shapely_final, layer=MT1)
#    # c.add_polygon(P_shapely_final, layer=MT1)
#
#     Tr1 = [(c.xmin-5, c.ymin-5), (c.xmin-5, c.ymax+5), (c.xmax+5, c.ymax+5), (c.xmax+5, c.ymin-5)]
#     # x_points = [-20, -20, 20, 20]
#     # y_points = [-20, 20, 20, -20]
#     c.add_polygon(Tr1, layer=MT1_DUMMY_BLOCK)
#     return c
#
# c = gf.Component("fill test")
# ref_Metal = c << a()
# fill_size = [0.8, 0.8]
# layers = [VIA1]
# c << gf.fill_rectangle(
#     ref_Metal,
#     fill_size=fill_size,
#     fill_layers=[VIA1],
#     margin=1,
#     fill_densities=[0.191]*len(layers),
#     avoid_layers=[MT1_DUMMY_BLOCK],
#     include_layers=[MT1],
#
# )
# c.show()
# #


# c = gf.Component("wsclDOEcell")
# combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_taper": True, "DC_MT1": False, "RF_out": True, "termination": 84, "s2s_type": "adiabatic",
#                     "s2s_O_len_OXOP": 50,
#                    "PS_sipho_length": 750,
#                    "PS_length": 850,
#                    "trans_length": 200,
#                    "trans_length_to_term": 200,
#                     "PS_trans_length": 100,
#                    "htr_bridge_place": True,
#                    "htr_bridge_width": 4,
#                    "htr_bridge_pitch": 70,
#                    "htr_bridge_reps": 3,

#                    "pad_center_gnd_width": 65,
#                    "pad_inner_gap_width": 35,
#                    "pad_sig_width": 65,
#                    "pad_outer_gap_width": 35,
#                    "pad_outer_gnd_width": 65,  # 200-7.5,
#                    "pad_length": 60,

#                    "S2S_center_gnd_width": 50,
#                    "S2S_inner_gap_width": 16,
#                    "S2S_sig_width": 40,
#                    "S2S_outer_gap_width": 5,
#                    "S2S_outer_gnd_width": 60,
#                    "S2S_length": 50,

#                    "PS_center_gnd_width": 50,
#                    "PS_inner_gap_width": 16,
#                    "PS_sig_width": 40,
#                    "PS_outer_gap_width": 15,  # used for GSGSG only
#                    "PS_outer_gnd_width": 50,  # -40, #used for GSGSG only

#                    }
# # temp fixes and offsets
# combined_params["wsclDOEcell_placeGC"] = 1
# _ =c << wsclDOEcell(combined_params)
# c.show()


#
# c = gf.Component("s2s_adiabatic_sample_2025_12_11_v1")
# combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True,
#                    "W": 0.4,
#                    "T": 0.3,
#                    "W0": 0.18,
#                    "R": 0.2,
#                    "S": 0.17,
#                    "G": 0.17,
#                    "L1": 4,
#                    "L2": 4,
#                    "L3": 4,
#                    "B1": 2.25,
#                    "B2": 0.87,
#                    "B3": 5,
#                    "C1": 0.525,
#                    "C2": 0.13 #+ 0.04,
#             }
# # combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True,
# #                    "W": 0.4,
# #                    "T": 0.3,
# #                    "W0": 0.18,
# #                    "R": 0.18,
# #                    "S": 0.18,
# #                    "G": 0.18,
# #                    "L1": 4,
# #                    "L2": 4,
# #                    "L3": 4,
# #                    "B1": 2.25,
# #                    "B2": 0.87,
# #                    "B3": 2,
# #                    "C1": 0.525,
# #                    "C2": 0.13 + 0.04,
# #             }
# _ = c << s2s_adiabatic(combined_params)
# #_.rotate(-90)
# c.show()


#
# c = gf.Component("test")
# b = gf.Component("resistor")
# r1 = gf.components.rectangle(size=(5, 5), layer=MT1)
# r2 = gf.components.rectangle(size=(5, 5), layer=MT1)
# r2.movex(2.5)
#
#
# b = gf.geometry.boolean(r1, r2, "and")
#
# p1 = gf.geometry.fillet(b, radius=1)
# c.add_polygon(p1)
#

# c.show()



#
# combined_params["PS_length"] = 1800
# c = gf.Component("OCDR_DOEcell_new")
# DOEocdr1=c<<OCDR_DOEcell_from_csv_params(combined_params, "DOE6_CSV_Params/OCDR_DOE1_pars.csv")
# c.show()


































#Altair layout test
# combined_DOE8_params = {**DOE8_params, **electrode_DOE8_params, "w_slot": 0.18, "w_OXOP": 2, "PS_length": 1200, "gap_IND_WG": 1,
#                     "sig_trace": "narrow", "taper_type": 1}


# c = gf.Component("PS_connected")
# _ = c << PS_connected_from_params(combined_DOE8_params)

# c = gf.Component("PS_slotWG")
# _ = c << PS_slotWG_from_params(combined_Vega_params, electrical=True)

# c = gf.Component("s2s_AIM")
# _ = c << s2s_from_params(combined_params)

# c = gf.Component("GSG_test_straight")
# _ = c << GSG_test_straight(pad_center_gnd_width=94, pad_inner_gap_width=6, pad_length = 100, test_length=1840, params=combined_DOE8_params)
#
# c = gf.Component("GSGSG_test_straight")
# _ = c << GSGSG_test_straight(pad_center_gnd_width=94, pad_inner_gap_width=6, pad_sig_width = 94, pad_outer_gap_width = 6, pad_outer_gnd_width = 94,  pad_length = 100, test_length=1840, params=combined_DOE8_params)

# c = gf.Component("GSG_test_group")
# _ = c << GSG_test_group_shared_gnd("GSG_Group_1_pars.csv", num_pairs = 5)

# c = gf.Component("MZM_GSGSG")
# _ = c <<MZM_GSGSG_from_params(params=combined_DOE8_params, trans_length=200, taper_type=1, sig_trace="narrow", termination=35, config="standard", gsgsg_variant="DC")

# c = gf.Component("wscl2_new")
# _ = c << wscl2()

# c = gf.Component("GSG_DOE")
# _ = c << GSG_DOE({**electrode_DOE1_params, "taper_type": 3})

# c = gf.Component("GSG_DOE")
# _ = c << GSGSG_DOE({**electrode_DOE1_params, "taper_type": 2, "sig_trace": "narrow"})

# #c = gf.Component("TC_DOE")
# _ = c << TC_DOE_s2s_only({**DOE5_params, **electrode_DOE5_params}, "Test_Dev_CSV_Params/TC_DOE_pars.csv", 1)
# _.move((-100, 0))
#
#
# #c = gf.Component("fringe_cap_doe_cell")
# _ = c << fringeCapDOEcell(n_fingers=15, len_finger=150, w_finger=6, gap_finger=10, RFpad_y_offset=-100)


# die_wscl_1 = c << relia()
#ie_wscl_1.move(( 0, 11100))
#
# die_DOE1 = c << DOE1()
# die_DOE1.move(( 12000, 11100 +3800*4 ))



#combined_params = {**DOE6_params, **electrode_DOE6_params, "PS_length":1000}
#_ = c << MZM_SilTerra(combined_params_new)


# combined_params_new = {**DOE6_params, **electrode_DOE6_params, **MZM_sipho_params}
# c = gf.Component("place_array_test")
#
# array, array_refs = PEO_place_array(MZM_SilTerra, combined_params_new, "Test_Dev_CSV_Params/MZM_GSGSG_pars.csv", n_rows=5, n_cols=5, pitch_x=1000, pitch_y=3000, rotation=-90)
# _ = c << array
# print(array_refs[0])
# _ = c << array
# _.move((0, -5000))

# c = gf.Component("trombone")
# _ = c << trombone(100)

# c.show()
























#old layout_test
# import gdsfactory as gf
# from functools import partial
# import math
# from library_sipho import *
# from library_sipho_DOE1 import *
# from library_sipho_DOE8 import *
# from library_sipho_test_structures import *
# from library_EC import *
#
# from layout_die_DOE8 import *
#
# from library_electrode import *
# from library_electrode_DOE3 import *
# from library_electrode_DOE8 import *
# from library_electrode_test_structures import *
# from layer_info import *
# from PDK_AMF import *
#
# c = gf.Component("layout")
#
# # _ = c << PS_connected(1, 0.15, 2, 500)
# # _ = c << PS_connected_v6(0.2, 0.27, 3, 500, 2)
# # _ = c << MZM_GSGSG_DC_v6(w_slot=0.18, w_slotWG=0.27, w_OXOP=3, PS_length=1200, gap_IND_WG=1, trans_length=200,
# #                     taper_type=1, sig_trace="narrow")
# # _ = c << DOE8_MZM_GSGSG_DC(6)
# # _ = c << GSG_MT2(500, 200)
# # _ = c << MMI_1X2_bend_s()
# # _ = c << MZM_GSG(1, 0.15, 2, 500, 500)
# # _ = c << MZM_GSGSG_GC(1, 0.15, 2, 500, 500)
# # _ = c << DC_pads()
# # _ = c << WG_straight(length=75)
# # _ = c << Frame()
# # _ = c << dam_array_DOE()
# # _ = c << WG_pair_delta_L(delta_L=135),
# # _ = c << gf.components.pad(size=(20, 0.1), layer=MT2, port_inclusion=0)
# # _ = c << Nano3_litho_marks()
# # _ = c << s2s_extra_slab(w_slot=0.2, w_OXOP=3)
# # _ = c << GSGSG_MT2_term(PS_length, trans_length, taper_type=1, termination=35, sig_trace="narrow")
# # _ = c << DC_2X2_FNG_v1(20)
# # _ = c << DC_2X2_FNG_v2(20)
# # _ = c << DC_FNG_v1_cascade6(8, 13, 34, 48, 106, 178)
# # _ = c << SGS_MT2_DC(PS_length=1000, trans_length=200, taper_type=1, sig_trace="narrow")
# # _ = c << WG_term(20)
# # _ = c << s2s(0.2, 3)
# # _ = c << WG_elec_connect()
# # _ = c << EC_array_gap(N=8, pitch=127, n=2)
# # _ = c << SGS_MT2_DC_batch(PS_length=1000, trans_length=200, taper_type=1, sig_trace="narrow")
# # _ = c << gf.components.straight(length=PS_length, width=10, layer=OXOP)
# # _ = c << GSG_test_straight(pad_cen_gnd_w=120, pad_in_gap_w=34, pad_s_w=120,
# #                         pad_out_gap_w=34, pad_out_gnd_w=120, test_length=1000, pad_length=150)
# # _ = c << GSGSG_resistor(termination=35)
# # _ = c << Dektak(length=640, width=150, block_len=20, pitch=40)
# _ = c << EC_TriTip_loop(sec1_len=sec1_len, sec2_len=sec2_len, sec3_len=sec3_len,
#                         w1=w1, w2=w2, w3=w3, w3_side=w3_side, w4=w4,
#                         pitch1=pitch1, pitch2=pitch2, pitch3=pitch3)
#
# # _ = c << SEM_160nm_2um()
#
#
# # die_wscl_1 = c << gf.import_gds("wscl1_GLM.gds", "wscl1", with_metadata=True)
# # die_wscl_1.move(( 0, 11100))
#
# # die_DOE5 = c << gf.import_gds("DOE5_0109.gds", "layout_die_DOE5_MZM_terminated", with_metadata=True)
# # die_DOE5.move(( 12000, 11100 ))
#
#
# c.show()
#
# # c.plot()
#
# _.pprint_ports()
#
# # _.pprint_ports()
#
# # _.pprint()

