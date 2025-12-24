"""
Unified parameter definitions for all DOE versions.
This file contains all the different parameter sets used across DOE1-8.

   Parameters:
    - w_slot: slot width
    - w_OXOP: oxide opening width for PS
    - PS_length: phase shifter length
    - gap_IND_WG: gap between inductor and waveguide
    - trans_length: transition length
    - taper_type: taper type
    - sig_trace: signal trace parameter
    - w_slotWG: slot waveguide width (defaults to global constant if None)
    - gap_NCONT_WG: gap between N-contact and waveguide (defaults to global constant if None)
    - gap_si_contact: gap between silicon contact and waveguide (defaults to global constant if None)
    - buffer_RIB_SLAB_overlay: slab overlay buffer (defaults to global constant if None)
    - s2s_type: type of s2s transition - "FNG", "oxide", or "extra slab" (defaults to "FNG")
    - s2s_w_OXOP: oxide opening width for s2s (defaults based on s2s_type if None)
    - electrode_type: type of electrode - "standard", "batch", or "compact" (defaults to "standard")


Table of Contents:
shared params
DOE1-DOE8
Relia
wscl

"""
MZM_config_default_sipho_params = { # type override parameters needed for MZM_GSGSG not in Altair sipho params
    "trans_length": 200,
    "taper_type": 2,  # 1 or 2
    "sig_trace": "narrow", # "narrow", "medium", or "wide"
    "termination": 0,  # 35, 50, or 65 (ohms) or 0 to match input pads
    "extra_slab": False,
    "config": "standard",  # "standard", "batch", "compact"
    "gsgsg_variant": "default",  # or "S21", "DC", "sym"
    "Oband_variant": False,
    "s2s_type": "MMI", #can be MMI or adiabatic
    "ps_config": "default",
    "gnds_shorted": False,
}

s2s_Oband_default_sipho_params = {#default for SYSTEM DIE 1, leave as is
    # s2s parameters - O-band specific
    "s2s_O_len_in": 0,  # Different
    "s2s_O_w_in": 0.38,  # Different
    "s2s_O_len_MMI": 0.96,# Different
    "s2s_O_w_MMI": 0.96, # Different
    "s2s_O_MMI_METCH_buffer": 0.1,
    "s2s_O_METCH_taper_buffer": 0.1,
    "s2s_O_len_taper": 80,#6,
    "s2s_gap_MMI_extra_slab": 0.4,
    "s2s_slot_start_width": 0.13,
    "s2s_O_len_OXOP": 0
}

s2s_adiabatic_default_sipho_params = {
    "S2S_ADIA_W": 0.38,
    "S2S_ADIA_T": 0.3,
    "S2S_ADIA_W0": 0.18,
    #"S2S_ADIA_R": 0.2,
    #"S2S_ADIA_S": 0.17,
    "S2S_ADIA_G": 0.17,
    "S2S_ADIA_L1": 4,
    "S2S_ADIA_L2": 4,
    "S2S_ADIA_L3": 42,
    "S2S_ADIA_B1": 2.25,
    "S2S_ADIA_B2": 0.87,
    "S2S_ADIA_B3": 5,
    "S2S_ADIA_C1": 0.525,
    "S2S_ADIA_C2": 0.13
}

s2s_AIM_sipho_params = { #legacy, may delete later
    "w_slab_s2s": 4,
    "w_RY_min": 0.6,
    "w_RY_outside_SE": 0.3,
    "w_inner_slab": 10,
}

ps_balun_sipho_params = { #default for SYSTEM DIE 1, leave as is
    "PS_length": 500,
    "trans_length": 250,
    "trans_length_to_term": 250,

    # slot WG parameters
    "w_slot": 0.16,  # Different
    "w_slotWG": 0.18,  # Different
    "w_slab": 4,
    "buffer_RIB_SLAB_overlay": 0.07,
    "w_OXOP": 5,

    "w_FETCH_CLD": 60,

    # doping regions
    "w_NIM": 6,
    "len_NIM": 5,
    "delta_to_len_taper_NIM": -0.5,
    "w_IND": 12.5,
    "gap_IND_WG": 1,
    "w_NCONT": 11.5,
    "gap_NCONT_WG": 2,
    "sb_overplot": 0.5,

    # conductive path structures

    "w_si_contact": 13.5,  # 10.5,
    "gap_si_contact": -2.25,
   # "gap_silicide":,
    "silicide_width": 3,
    "gap_silicide": 0.55,
    "electrical": True,
    "num_rows_CONTACT": 3,
    "via_size_contact": (0.8, 0.8),
    "gap_via_contact": 0.6,

    "extension_electrical": 3,

    "w_MT1": 5,#13,
    "min_gap_OXOP_MT": 5,
    "num_rows_V1": 3,
    "via_size_1": 0.8,#unused in 2 layer setup
    "gap_via_1": 0.6,
    "slot_size_MT1": (1.5, 1.5),
    "slot_gap_MT1": 10,
    

    "w_MT2": 93,  # 10,
    "min_inc_via_1": 0.1, #CHECK
    "num_rows_V2": 2,
    "min_exc_of_via_2": 0.2,#CHECK

    "via_size_top": 0.35,
    "gap_via_top": 0.45,
}

ps_SilTerra_sipho_params = { 
    "PS_length": 1500,
    "trans_length": 250,
    "trans_length_to_term": 250,

    # slot WG parameters
    "w_slot": 0.2,  # Different
    "w_slotWG": 0.2,  # Different
    "w_routing": 0.4,
    "w_slab": 4,
    "buffer_RIB_SLAB_overlay": 0.1,
    "w_OXOP": 5,

    "w_impl_window": 40,

    # doping regions
    "w_NIM": 6,
    "len_NIM": 5,
    "delta_to_len_taper_NIM": -0.5,
    "w_IND": 12.5,
    "gap_IND_WG": 1,
    "w_NCONT": 11.5,
    "gap_NCONT_WG": 2,
    "sb_overplot": 0.5,

    # conductive path structures

    "w_si_contact": 9.5,  # 10.5,
    "gap_si_contact": -2.25,
   # "gap_silicide":,
    "silicide_width": 3.5,
    "gap_silicide": 0.55,
    "electrical": True,
    "num_rows_CONTACT": 2,
    "via_size_contact": (0.8, 0.8),
    "gap_via_contact": 0.6,

    "extension_electrical": 3,

    "w_MT1": 20,#13,
    "min_gap_OXOP_MT": 5,
    "num_rows_V1": 3,
    "via_size_1": 0.8,#unused in 2 layer setup
    "gap_via_1": 0.6,
    "slot_size_MT1": (1.5, 1.5),
    "slot_gap_MT1": 10,
    

    "w_MT2": 20,  # 10,
    "min_inc_via_1": 0.1, #CHECK
    "num_rows_V2": 2,
    "min_exc_of_via_2": 0.2,#CHECK

    "via_size_top": 0.8,
    "gap_via_top": 0.6,
}


balun_sipho_params = {
    **MZM_config_default_sipho_params,
    "Oband_variant": True,
    **s2s_Oband_default_sipho_params,
    **s2s_AIM_sipho_params,
    **s2s_adiabatic_default_sipho_params,
    **ps_balun_sipho_params,

    "s2s_sipho_length": s2s_Oband_default_sipho_params["s2s_O_len_in"] + s2s_Oband_default_sipho_params["s2s_O_len_MMI"] + s2s_Oband_default_sipho_params["s2s_O_MMI_METCH_buffer"]
    + s2s_Oband_default_sipho_params["s2s_O_METCH_taper_buffer"] + s2s_Oband_default_sipho_params["s2s_O_len_taper"],
    "ps_connected_total_length": 2*(s2s_Oband_default_sipho_params["s2s_O_len_in"] + s2s_Oband_default_sipho_params["s2s_O_len_MMI"] + s2s_Oband_default_sipho_params["s2s_O_MMI_METCH_buffer"]
    + s2s_Oband_default_sipho_params["s2s_O_METCH_taper_buffer"] + s2s_Oband_default_sipho_params["s2s_O_len_taper"])
    + ps_balun_sipho_params["PS_length"] + ps_balun_sipho_params["trans_length"] + ps_balun_sipho_params["trans_length_to_term"]
}

SilTerra_sipho_params = {
    **MZM_config_default_sipho_params,
    "Oband_variant": True,
    **s2s_Oband_default_sipho_params,
    **s2s_AIM_sipho_params,
    **s2s_adiabatic_default_sipho_params,
    **ps_SilTerra_sipho_params,

    "s2s_sipho_length": s2s_Oband_default_sipho_params["s2s_O_len_in"] + s2s_Oband_default_sipho_params["s2s_O_len_MMI"] + s2s_Oband_default_sipho_params["s2s_O_MMI_METCH_buffer"]
    + s2s_Oband_default_sipho_params["s2s_O_METCH_taper_buffer"] + s2s_Oband_default_sipho_params["s2s_O_len_taper"],
    "ps_connected_total_length": 2*(s2s_Oband_default_sipho_params["s2s_O_len_in"] + s2s_Oband_default_sipho_params["s2s_O_len_MMI"] + s2s_Oband_default_sipho_params["s2s_O_MMI_METCH_buffer"]
    + s2s_Oband_default_sipho_params["s2s_O_METCH_taper_buffer"] + s2s_Oband_default_sipho_params["s2s_O_len_taper"])
    + ps_balun_sipho_params["PS_length"] + ps_balun_sipho_params["trans_length"] + ps_balun_sipho_params["trans_length_to_term"]
}

SYSTEM_1_params = {
    **balun_sipho_params
}

# DOE1 Parameters
DOE1_params = {
    "DOE_name": "DOE1",  # Add identifier for electrode parameter selection
    # slot WG parameters
    "w_slot": 0.15,
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 1000,
    "buffer_RIB_SLAB_overlay": 0.09,
    
    # PS structures
    "w_OXOP": 3,
    "w_si_contact": 7,
    "gap_si_contact": 3.5,
    
    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 5,
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM" : -0.5,
    
    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,
    
    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,
    
    # PS types
    "PS_type": 1,
    
    # LC droplet boundary
    "LC_bound_width": 50,
    
    # Frame parameters
    "frm_w": 6400,  # Corrected to match original DOE1 layout
    "frm_len": 3800,  # Corrected to match original DOE1 layout
    "trench_w": 100,
    
    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    
    "MMI_delta_L": 70,
    "Si_WG_pitch": 5,  # Corrected to match original DOE1 layout
}

# DOE2 Parameters (similar to DOE1 but with specific differences)
DOE2_params = {
    "DOE_name": "DOE2",  # Add identifier for electrode parameter selection
    # slot WG parameters - DOE2 specific values
    "w_slot": 0.2,  # Different from DOE1 (0.15)
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 1000,
    "buffer_RIB_SLAB_overlay": 0.09,
    
    # PS structures
    "w_OXOP": 3,
    "w_si_contact": 7,
    "gap_si_contact": 3.5,
    
    # doping regions - DOE2 specific values
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 5,  # Should match original DOE1 library value, not layout file hard-coded 2.5
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM" : -0.5,
    
    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,
    
    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,
    
    # PS types
    "PS_type": 1,
    
    # LC droplet boundary
    "LC_bound_width": 50,
    
    # Frame parameters (same as DOE1)
    "frm_w": 6400,
    "frm_len": 3800,
    "trench_w": 100,
    
    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    
    "MMI_delta_L": 70,
    "Si_WG_pitch": 5,
}

# DOE3 Parameters 
DOE3_params = {
    "DOE_name": "DOE3",  # Add identifier for electrode parameter selection
    # slot WG parameters
    "w_slot": 0.15,
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 500,  # Different from DOE1
    "buffer_RIB_SLAB_overlay": 0.09,
    
    # PS structures
    "w_OXOP": 3,
    "w_si_contact": 7,
    "gap_si_contact": 3.5,
    
    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 5,
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM" : -0.5,
    
    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,
    
    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,
    
    # PS types
    "PS_type": 1,
    
    # LC droplet boundary
    "LC_bound_width": 50,
    
    # Frame parameters
    "frm_w": 25000,
    "frm_len": 30300,
    "trench_w": 100,
    
    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    
    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,
}

# DOE4 Parameters
DOE4_params = {
    "DOE_name": "DOE4",  # Add identifier for electrode parameter selection
    # slot WG parameters
    "w_slot": 0.15,
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 1000,  # Changed from 500 to match original DOE4
    "buffer_RIB_SLAB_overlay": 0.09,
    
    # PS structures
    "w_OXOP": 3,
    "w_si_contact": 7,
    "gap_si_contact": 3.5,
    
    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 5,  # Corrected back to 5 to match original DOE4 library
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM" : -0.5,
    
    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,
    
    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,
    
    # PS types
    "PS_type": 1,
    
    # LC droplet boundary
    "LC_bound_width": 50,
    
    # Frame parameters
    "frm_w": 6400,  # Different
    "frm_len": 3800,  # Different
    "trench_w": 100,
    
    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    
    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,
}

# DOE5 Parameters (O-band specific)
DOE5_params = {
    "DOE_name": "DOE5",  # Add identifier for electrode parameter selection
    # slot WG parameters
    "w_slot": 0.16,  # Different
    "w_slotWG": 0.24,  # Different
    "w_slab": 4,
    "PS_length": 1000,
    "buffer_RIB_SLAB_overlay": 0.09,
    
    # PS structures
    "w_OXOP": 2,  # Different
    "w_si_contact": 7,
    "gap_si_contact": 3.5,
    
    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 2.5,  # Different
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM": -0.5,

    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,
    
    # s2s parameters - O-band specific
    "s2s_O_len_in": 2.1,  # Different
    "s2s_O_w_in": 0.41,  # Different  
    "s2s_O_len_MMI": 1.30,  # Different
    "s2s_O_w_MMI": 1.06,  # Different
    "s2s_O_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,
    
    # PS types
    "PS_type": 1,
    
    # LC droplet boundary
    "LC_bound_width": 50,
    
    # Frame parameters
    "frm_w": 25000,
    "frm_len": 30300,
    "trench_w": 100,
    
    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    
    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,
}

DOE6_params = {
    **balun_sipho_params,
    "DOE_name": "DOE6",
    # slot WG parameters
    "w_slot": 0.15,
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 500,
    "buffer_RIB_SLAB_overlay": 0.09,

    # PS structures
    "w_OXOP": 3,
    "w_si_contact": 7,
    "gap_si_contact": 3.5,

    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 5,
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM": -0.5,

    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,

    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,

    # PS types
    "PS_type": 1,

    # LC droplet boundary
    "LC_bound_width": 50,

    "frm_w": 25000,
    "frm_len": 30300,
    "trench_w": 100,

    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,

    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,

    "w_FETCH_CLD": 60
}

# DOE8 Parameters
DOE8_params = {
    "DOE_name": "DOE8",  # Add identifier for electrode parameter selection
    # slot WG parameters
    "w_slot": 0.15,
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 500,
    "buffer_RIB_SLAB_overlay": 0.09,
    
    # PS structures
    "w_OXOP": 5,  # Different
    "w_si_contact": 9,  # Different
    "gap_si_contact": 3.5,
    
    # doping regions
    "w_NCONT": 10,  # Different
    "gap_NCONT_WG": 2.5,
    "w_IND": 5,
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM": -0.5,
    
    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,
    
    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,
    
    # PS types
    "PS_type": 1,
    
    # LC droplet boundary
    "LC_bound_width": 50,
    
    # Frame parameters
    "frm_w": 25000,
    "frm_len": 30300,
    "trench_w": 100,
    
    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    
    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,
    "trans_length": 200,
}

relia_params = {
    "DOE_name": "relia",
    # slot WG parameters
    "w_slot": 0.2,
    "w_slotWG": 0.31,
    "w_slab": 4,
    "PS_length": 1000,
    "buffer_RIB_SLAB_overlay": 0.09,

    # PS structures
    "w_OXOP": 3,
    "w_si_contact": 7,
    "gap_si_contact": 3,

    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 2.5,
    "w_IND": 2.5,
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM": -0.5,

    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,

    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,

    # PS types
    "PS_type": 1,

    # LC droplet boundary
    "LC_bound_width": 50,

    "frm_w": 25000,
    "frm_len": 30300,
    "trench_w": 100,

    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,

    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,
}


wscl2_params = {
    "DOE_name": "wscl2",
    # slot WG parameters
    "w_slot": 0.2,
    "w_slotWG": 0.27,
    "w_slab": 4,
    "PS_length": 1440,
    "buffer_RIB_SLAB_overlay": 0.09,

    # PS structures
    "w_OXOP": 3,
    "w_IND_WG": 18,
    "w_si_contact": 7,
    "gap_si_contact": 3.5,

    # doping regions
    "w_NCONT": 8,
    "gap_NCONT_WG": 3,
    "w_IND": 2.5,
    "gap_IND_WG": 1,
    "w_NIM": 6,
    "delta_to_len_taper_NIM": -0.5,

    # conductive path structures
    "w_MT1": 10.5,
    "min_gap_OXOP_MT": 5,
    "via_size": 3,
    "gap_via_1": 2,
    "min_inc_via_1": 0.5,
    "min_exc_of_via_2": 2,

    # s2s parameters
    "s2s_len_in": 2.02,
    "s2s_width_in": 0.5,
    "s2s_len_MMI": 1.38,
    "s2s_width_MMI": 1.3,
    "s2s_len_taper": 6,
    "s2s_gap_MMI_extra_slab": 0.4,

    # PS types
    "PS_type": 1,

    # LC droplet boundary
    "LC_bound_width": 50,

    "frm_w": 25000,
    "frm_len": 30300,
    "trench_w": 100,

    # Dam array parameters
    "dam_w": 1000,
    "dam_len": 1500,
    "dam_wall_w": 100,
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,

    "MMI_delta_L": 70,
    "Si_WG_pitch": 10,
}

SEM_params = {**DOE8_params, "w_OXOP": 18, "w_si_contact": 7, "gap_si_contact": 3.5, "w_IND_WG": 18}
# # w_OXOP = 3
# # w_IND_WG = 18
# # w_si_contact = 7
# # gap_si_contact = 3.5
#     "DOE_name": "SEM",
#     # slot WG parameters
#     "w_slot": 0.15,
#     "w_slotWG": 0.27,
#     "w_IND_WG": 18,
#     "w_slab": 4,
#     "PS_length": 1000,
#     "buffer_RIB_SLAB_overlay": 0.09,
#
#     # PS structures
#     "w_OXOP": 3,
#     "w_si_contact": 7,
#     "gap_si_contact": 3.5,
#
#     # doping regions
#     "w_NCONT": 8,
#     "gap_NCONT_WG": 3,
#     "w_IND": 2.5,
#     "gap_IND_WG": 1,
#     "w_NIM": 6,
#     "delta_to_len_taper_NIM": -0.5,
#
#     # conductive path structures
#     "w_MT1": 10.5,
#     "min_gap_OXOP_MT": 5,
#     "via_size": 3,
#     "gap_via_1": 2,
#     "min_inc_via_1": 0.5,
#     "min_exc_of_via_2": 2,
#
#     # s2s parameters
#     "s2s_len_in": 2.02,
#     "s2s_width_in": 0.5,
#     "s2s_len_MMI": 1.38,
#     "s2s_width_MMI": 1.3,
#     "s2s_len_taper": 6,
#     "s2s_gap_MMI_extra_slab": 0.4,
#
#     # PS types
#     "PS_type": 1,
#
#     # LC droplet boundary
#     "LC_bound_width": 50,
#
#     "frm_w": 25000,
#     "frm_len": 30300,
#     "trench_w": 100,
#
#     # Dam array parameters
#     "dam_w": 1000,
#     "dam_len": 1500,
#     "dam_wall_w": 100,
#     "dam_N": 9,
#     "dam_pitch_w": 2500,
#     "dam_pitch_len": 2750,
#
#     "MMI_delta_L": 70,
#     "Si_WG_pitch": 10,
# }

# Parameter map for easy access
DOE_PARAMS = {
    "DOE1": DOE1_params,
    "DOE2": DOE2_params,
    "DOE3": DOE3_params,
    "DOE4": DOE4_params,
    "DOE5": DOE5_params,
    "DOE6": DOE6_params,
    "DOE8": DOE8_params,
}

# Additional common parameters (if needed)
common_params = {
    "dam_N": 9,
    "dam_pitch_w": 2500,
    "dam_pitch_len": 2750,
    "MMI_delta_L": 70,
    "Si_WG_pitch": 10
}

