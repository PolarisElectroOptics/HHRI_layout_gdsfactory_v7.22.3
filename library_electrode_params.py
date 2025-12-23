# Shared base structures
from layer_info import MT2, PAD, HTR, VIA2

pad_params = {
    "pad_center_gnd_width": 60,
    "pad_inner_gap_width": 30,
    "pad_sig_width": 80,
    "pad_outer_gap_width": 30,
    "pad_outer_gnd_width": 102.5,#200-7.5,
    "pad_length": 60,
}

s2s_electrode_params = {
    "S2S_center_gnd_width": 200,
    "S2S_inner_gap_width": 13,
    "S2S_sig_width": 140,
    "S2S_outer_gap_width": 10,
    "S2S_outer_gnd_width": 140,
    "S2S_length": 10,
}
s2s_electrode_balun_params = {
    "S2S_center_gnd_width": 20,
    "S2S_inner_gap_width": 3.5,
    "S2S_sig_width": 101.25,#102.45,
    "S2S_outer_gap_width": 15,
    "S2S_outer_gnd_width": 140,
    "S2S_length": 16,#30,
    }

s2s_electrode_differential_params = {
    "S2S_center_gnd_width": 50,
    "S2S_inner_gap_width": 16,
    "S2S_sig_width": 37,
    "S2S_outer_gap_width": 8,
    "S2S_outer_gnd_width": 231- 44.5,
    "S2S_length": 50,
}


ps_electrode_params = {
    "PS_center_gnd_width": 200,
    "PS_inner_gap_width": 13,
    "PS_sig_width": 10,
    "PS_outer_gap_width": 140,
    "PS_outer_gnd_width": 50,

    "PS_MT1_center_gnd_width": 3.5,
    "PS_MT1_inner_gap_width": 20,
    "PS_MT1_sig_width": 93,
    "PS_MT1_outer_gap_width": 0,
    "PS_MT1_outer_gnd_width": 0,
}
ps_electrode_narrow_params = {
    "PS_center_gnd_width": 200,
    "PS_inner_gap_width": 13,
    "PS_sig_width": 0,
    "PS_outer_gap_width": 140,
    "PS_outer_gnd_width":50
}
ps_electrode_balun_params = {
    "MT1_from_PS": False,
    "PS_center_gnd_width": 20,
    "PS_inner_gap_width": 3.5,#13 + 20 + 3.5/2,
    "PS_sig_width": 80+ 21.25,
    "PS_outer_gap_width": 35, #used for GSGSG only
    "PS_outer_gnd_width": 100+6, #used for GSGSG only

    "PS_MT1_center_gnd_width": 3.5,
    "PS_MT1_inner_gap_width": 20,
    "PS_MT1_sig_width": 93,
    "PS_MT1_outer_gap_width": 0,
    "PS_MT1_outer_gnd_width": 0,
}

ps_electrode_differential_params = {
    #"PS_length": 500, #port to here from sipho params when possible

    "MT1_from_PS": True, #places grounds from PS_slotWG rather than central one in GSGSG_MT2
    "PS_center_gnd_width": 50,
    "PS_inner_gap_width": 16,
    "PS_sig_width": 32,
    "PS_outer_gap_width": 28, #used for GSGSG only
    "PS_outer_gnd_width": 165.5,#-40, #used for GSGSG only

    "PS_trans_length": 100,

    # "PS_MT1_center_gnd_width": 50,
    # "PS_MT1_inner_gap_width": 16,
    # "PS_MT1_sig_width": 32,
    # "PS_MT1_outer_gap_width": 28,
    # "PS_MT1_outer_gnd_width": 165.5,# - 40,



    # "pad_MT1_center_gnd_width": 0,
    # "pad_MT1_inner_gap_width": 25 + 74,
    # "pad_MT1_sig_width": 2,
    # "pad_MT1_outer_gap_width": 98 ,
    # "pad_MT1_outer_gnd_width": 2,


}

pad_term_params = {
    # "pad_t_center_gnd_width": 100,
    # "pad_t_inner_gap_width": 20,
    # "pad_t_sig_width": 80,
    # "pad_t_outer_gap_width": 20,
    # "pad_t_outer_gnd_width": 86,
    "pad_t_length": 60,
}

pad_term_bond_params = {
    # metal geometric parameters for bonding (wide design)
    "pad_t_center_gnd_width":120,
    "pad_t_inner_gap_width":34,
    "pad_t_sig_width":120,
    "pad_t_outer_gap_width":34,
    "pad_t_outer_gnd_width":120,
    "pad_t_length":400
}

heater_params = {
    "htr_length": 104,
    "htr_width_35": 8.562,
    "htr_width_50": 5.995,
    "htr_width_65": 4.613,
    "htr_width_84": 15,

    #new meandering heater params
    #"closer" will be the params closer to the termination pads
    "htr_further_center_gnd_width": 0,
    "htr_further_inner_gap_width": 20,
    "htr_further_sig_width": 2*30+20,
    "htr_further_outer_gap_width": 0,
    "htr_further_outer_gnd_width": 0,

    "htr_closer_center_gnd_width": 100,
    "htr_closer_inner_gap_width": 20,
    "htr_closer_sig_width": 45 + 10,
    "htr_closer_outer_gap_width": 0,
    "htr_closer_outer_gnd_width": 0,

    "htr_length": 130,
    "htr_width": 30,
    "htr_fillet_radius": 5,
    "htr_flipped": True,

    "htr_connect_length": 32,  # repurposed for M1 pads
    "htr_connect_width": 10,
    "htr_connect_fillet_radius": 1,

}

common_params = {
    "sc_length": 5,
    "layer_MT2": MT2,
    "layer_PAD": PAD,
    "layer_HTR": HTR,
    "layer_VIA2": VIA2,
    "spacing_x": 200,
    "spacing_y": 200,
    "DC_pad_size_x": 90,
    "DC_pad_size_y": 90,
    "pads_rec_gap": 30,

    "MT1_from_PS": True,
    "DC_MT1": False,
    "PS_taper": False,
    "MT_dummy_margin": 90,
}

# DOE-specific dictionaries

balun_electrode_params = { #based on DOE5 with modified spacings
    **common_params,
    **pad_params,
    **s2s_electrode_balun_params,
    **ps_electrode_balun_params,
    **pad_term_params,
    **heater_params,

    #AIM params
    # "slot_size_MT3": (10, 4), #unused..
    # "slot_gap_MT3": 10, #unused..
    #
    # "cheese": False,
    # "cheese_slot_size_MT3": (2, 2),
    # "gap_slot_MT3": 15,#30, #18 is good, 24 is ok
    # "border_slot_MT3": 10,
    # "cheese_shift_x": 0,
    # "cheese_shift_y": 0,
    # "center_slot_line_offset_MT3": -4,#7 #shift single line of slots in center of GSG transition up or down (and adjust length accordingly)
    # "center_slot_line_length_delta_MT3_term": 0,#-10,
    #
    # "cheese_slot_size_MT1": (1, 1),
    # "gap_slot_MT2": 10,

    "MT_dummy_margin": 90,
}

differential_electrode_params = {
    **pad_params,
    **s2s_electrode_differential_params,
    **ps_electrode_differential_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "MT_dummy_margin": 20,
    "RF_out": True
}



electrode_DOE1_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "PS_length": 1000,
    "trans_length": 425,
    "trans_length_to_term": 100,
    "DC_pad_size_y": 120.002,  # Override to match original DOE1 value
}

electrode_DOE3_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "PS_length": 500,
    "trans_length": 425,
    "trans_length_to_term": 200,
}

electrode_DOE4_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "PS_length": 500,
    "trans_length": 425,
    "trans_length_to_term": 100,
}

electrode_DOE5_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "PS_length": 500,
    "trans_length": 425,
    "trans_length_to_term": 100,
}

electrode_DOE6_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "trans_length": 450,
    "trans_length_to_term": 220,
    "DC_pad_gap": 30,
    "PS_length": 500,
}


electrode_DOE8_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "pad_center_gnd_width": 100,
    "pad_inner_gap_width": 20,
    "pad_sig_width": 80,
    "pad_outer_gap_width": 20,
    "pad_outer_gnd_width": 86,
    "pad_length": 160,
    "S2S_center_gnd_width": 200,
    "S2S_inner_gap_width": 13,
    "S2S_sig_width": 140,
    "S2S_outer_gap_width": 10,
    "S2S_outer_gnd_width": 140,
    "S2S_length": 10,
    "PS_center_gnd_width": 50,
    "PS_inner_gap_width": 20,
    "PS_sig_width": 20,
    "PS_outer_gap_width": 20,
    "PS_outer_gnd_width": 50,
    "PS_length": 1000,
    "pad_t_center_gnd_width": 100,
    "pad_t_inner_gap_width": 20,
    "pad_t_sig_width": 80,
    "pad_t_outer_gap_width": 20,
    "pad_t_outer_gnd_width": 86,
    "pad_t_length": 50,
    "htr_length": 50,
    "htr_width_35": 8.562,
    "htr_width_50": 5.995,
    "htr_width_65": 4.613,
    "htr_connect_length": 20,
    "sc_length": 10,
    "trans_length": 200,
    "layer_MT2": 125,
    "layer_PAD": 150,
    "layer_HTR": 115,
    "layer_VIA2": 120,
    "spacing_x": 200,
    "spacing_y": 200,
    "DC_pad_size_x": 110,
    "DC_pad_size_y": 150,
    "DC_pad_gap": 30,
    "pads_rec_gap": 30,  # Adding missing parameter for DOE8
    "trans_length_to_term": 100,
}

electrode_relia_params = {
    **pad_params,
    **pad_term_bond_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **heater_params,
    **common_params,
    "pad_length": 240,
    "PS_length": 1000,
    #"PS_total_width": PS_center_gnd_width + PS_inner_gap_width*2 + PS_sig_width*2 + PS_outer_gap_width*2 + PS_outer_gnd_width*2,
    # short circuit for ground pads
    "sc_length": 10,
    "trans_length": 790,
    "trans_length_to_term": 100,
    "DC_pad_size_x": 200,
    "DC_pad_size_y": 200,
    "DC_pad_gap": 50,
    "pads_rec_gap": 50
}

electrode_wscl2_params = {
    **pad_params,
    **s2s_electrode_params,
    **ps_electrode_params,
    **pad_term_params,
    **heater_params,
    **common_params,
    "pad_length": 100, #these are EES21 params
    "pad_center_gnd_width": 80,
    "pad_outer_gnd_width": 80,
    "S2S_length": 5,
    "PS_length": 1440,
    "trans_length": 95

}

