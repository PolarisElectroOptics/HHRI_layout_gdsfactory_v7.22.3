import gdsfactory as gf
from gdsfactory.cross_section import ComponentAlongPath
from functools import partial
import math
from layer_info import *
#from PDK_AMF_Oband import *
from PDK_Crealight import *
from floorplan import *

#from library_sipho import *
from library_sipho_params import *
from library_electrode_unified import *
import library_electrode_params



from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.cross_section import CrossSection, cross_section #xsection
from gdsfactory.typings import LayerSpec

import random #both these used only in DOE6 + 7 - pd is used for GLM's flow of reading from .csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #temp


'''Table of Contents:
- General Layout Functions
- Key Photonic Device PCells
- Test PCells
'''

'''GENERAL LAYOUT FUNCTIONS'''
#not specific to sipho, but placed here until a third library file is warranted
#@gf.cell #seems to be ok without decorator... removed it so a reference array can returned as well
def PEO_place_array(function, default_params, doe_CSV, n_rows:int, n_cols:int, pitch_x:float, pitch_y:float, name="array", rotation=0, start_line=2):
    """
    Generates array of any arbitrary PEO PCell that takes a parameter library as the sole input. Placement starts in bottom left and goes right and up.
    Provided CSV params override the provided default params.

    CSV format: first row is name of parameters
                following rows are parameter values - no blank rows allowed

    :param function:
    :param params: default parameters for a device, unless overridden in CSV
    :param doe_CSV: file path of CSV parameter overrides - if empty, function uses default parameters
    :param n_rows:
    :param n_cols:
    :param pitch_x: x pitch
    :param pitch_y: y pitch
    :param start_line: line in the csv document to start reading DOE data (=2 by default)
    :return: Component, array of references

    usage example:
    c = gf.Component()

    combined_params_new = {**DOE6_params, **electrode_DOE6_params, **MZM_sipho_params}
    array, array_refs = PEO_place_array(MZM_SilTerra, combined_params_new, "Test_Dev_CSV_Params/MZM_GSGSG_pars.csv", n_rows=5, n_cols=5, pitch_y=1000, pitch_x=3000)

    _ = c << array
    c.show()
    """
    rows_to_skip = np.arange(1, start_line-1)
    if doe_CSV:
        if start_line <= 2:
            doe = pd.read_csv(doe_CSV)
        else:
            doe = pd.read_csv(doe_CSV, skiprows=rows_to_skip)
    ref_array = []
    c = gf.Component(name)
    for i in range(n_rows): #per row
        for j in range(n_cols): #per cell
            combined_params = {**default_params}
            if doe_CSV:
                for col in doe.columns: #write new params from csv to code
                    combined_params[str(col)] = doe.iloc[i*(n_cols)+j].loc[col]
                    #print(doe.iloc[i*(n_cols)+j].loc["DUT"])
            #place object
            _ = c << function(params=combined_params)
            if "extra_rotation" in combined_params:
                _.rotate(combined_params["extra_rotation"])
            _.rotate(rotation).move((j*pitch_x, i*pitch_y))
            ref_array.append(_)

    return c, ref_array

'''KEY DEVICE PCells'''
#default CrossSection, used in MMI s-waveguides and some test structures
# @xsection
def rib_Oband(
        width: float = 0.4,
        layer: LayerSpec = (10, 0),
        radius: float = 15.0,
        radius_min: float = 5,
        width_clad: float = 7,
        **kwargs,
) -> CrossSection:
    """Return Strip cross_section."""
    s0=gf.Section(width=width, offset=0, layer=(10031, 1), port_names=("o1", "o2"))
    # layer (1031,1) is a help layer that will be replaced by (31,1) in _clad function
    s1 = gf.Section(width=width_clad * 2 + width, offset=0, layer=(10031, 2))
    # layer (1031,2) is a help layer that will be merged and replaced by layer (31,2) in merge_clad function
    ##not needed for now s2 = gf.Section(width=width_clad, offset=- width/2 - width_clad/2, layer=(131,2))

    x = gf.CrossSection(sections=(s0, s1), radius = radius, radius_min = radius_min)

    """Return Strip cross_section."""
    return x


#default CrossSection, used in MMI s-waveguides and some test structures
# @xsection

def rib(
        width: float = 0.4,
        layer: LayerSpec = (10, 0),
        radius: float = 15.0,
        radius_min: float = 5,
        width_clad: float = 6,
        **kwargs,
        
) -> CrossSection:



    s0 = gf.Section(width=width, offset=0, layer=(10031,1), port_names=("o1", "o2"))
    #layer (1031,1) is a help layer that will be replaced by (31,1) in merge_clad function 
    s1 = gf.Section(width=width_clad *2 + width, offset=0, layer=(10031,2))
    #layer (1031,2) is a help layer that will be merged and replaced by layer (31,2) in merge_clad function
    ##not needed for now s2 = gf.Section(width=width_clad, offset=- width/2 - width_clad/2, layer=(131,2))
    
    x = gf.CrossSection(sections=(s0, s1), radius = radius, radius_min = radius_min)
    
    """Return Strip cross_section."""
    return x


def metal1_metal2(
        params,
        width: float = 10.0,
        radius: float= 0.0,
        **kwargs,

) -> CrossSection:

    via_size_top = params["via_size_top"]
    gap_via_top = params["gap_via_top"]


    s0 = gf.Section(width=width, offset=0, layer=MT2, port_names=("e1", "e2"))
    s1 = gf.Section(width=width, offset=0, layer=MT1, port_names=("e1_m1", "e2_m1"))

    components_along_path = []
    num_rows_V1 = 10
    offset_via_1 = -num_rows_V1 * (via_size_top + gap_via_top) / 2 + gap_via_top
    for i in range(num_rows_V1):
        components_along_path.append(
            ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=5,
                               offset=(offset_via_1 + i * (via_size_top + gap_via_top))))

    x = gf.CrossSection(sections=(s0, s1), components_along_path=components_along_path)

    """Return Strip cross_section."""
    return x


@gf.cell
def merge_clad(c, rad_round = 2):
    
    #merge the FETCH_CLAD layer
    #input parameter  "c" is a component 
    #input parameter "rad_round" is rounding corner radius  
    #return a new component
    
    cc = c.copy()
    cc.name = "copy"
    
    #merge layer
    c_bool1= gf.geometry.boolean_klayout(cc, cc, operation = 'or', layer1 = (10031,2), layer2 = (10031,2), layer3 = (31,2))
    #extract polygon for apply rounding 
    ss= c_bool1.get_polygons((31,2))
    
    
    #rouding polygon
    for p in ss:
        p1 = cc.add_polygon(p, layer = RIB_ETCH)
        
        if (rad_round !=0 ):
            p_round = p1.fillet(rad_round,0.001)
        else :
            p_round = p1
        c.add_polygon(p_round, layer=RIB_ETCH)
        c.add_polygon(p_round, layer=NOP)

    """
    c_bool2 = gf.geometry.boolean_klayout(dd, dd, operation = 'not', layer1 = (31,2), layer2 = (131,1), layer3 = (31,2))
    ss= c_bool2.get_polygons((31,2))
    c.add_polygon(ss, layer = (31,2))
    """
    #remap help layer to functioning layer
    c.remap_layers({(10031,1):RIB})

    #remove help layer
    d = c.remove_layers(layers=((10031,2),))
    d.name = "potato"
    
    return d


@gf.cell
def Frame(frm_w, frm_len, trench_w):
    # def Frame():
    c = gf.Component()

    xpts = [0, frm_w, frm_w, 0]
    ypts = [0, 0, trench_w, trench_w]
    c.add_polygon([xpts, ypts], layer=DTR)

    xpts = [0, frm_w, frm_w, 0]
    ypts = [frm_len - trench_w, frm_len - trench_w, frm_len, frm_len]
    c.add_polygon([xpts, ypts], layer=DTR)

    xpts = [0, trench_w, trench_w, 0]
    ypts = [0, 0, frm_len, frm_len]
    c.add_polygon([xpts, ypts], layer=DTR)

    xpts = [frm_w - trench_w, frm_w, frm_w, frm_w - trench_w]
    ypts = [0, 0, frm_len, frm_len]
    c.add_polygon([xpts, ypts], layer=DTR)

    return c

@gf.cell
def s2s_from_params(params: dict, extra_slab: bool = False, Oband_variant: bool = False) -> gf.Component:
    """
    AMF version
    :param params:
    :param extra_slab:
    :param Oband_variant:
    :return:
    """
    c = gf.Component()
    # Unpack shared parameters
    w_slot = params["w_slot"]
    w_OXOP = params["w_OXOP"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    w_NIM = params["w_NIM"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]
    delta_to_len_taper_NIM = params["delta_to_len_taper_NIM"]
    extra_slab = params["extra_slab"]
    Oband_variant = params["Oband_variant"]

    s2s_slot_start_width = params["s2s_slot_start_width"]


    # Choose parameter names based on variant
    if Oband_variant:
        len_in = params["s2s_O_len_in"]
        width_in = params["s2s_O_w_in"]
        len_MMI = params["s2s_O_len_MMI"]
        width_MMI = params["s2s_O_w_MMI"]
        MMI_METCH_buffer=params["s2s_O_MMI_METCH_buffer"]
        METCH_taper_buffer=params["s2s_O_METCH_taper_buffer"]
        len_taper = params["s2s_O_len_taper"]
    else:
        len_in = params["s2s_len_in"]
        width_in = params["s2s_width_in"]
        len_MMI = params["s2s_len_MMI"]
        width_MMI = params["s2s_width_MMI"]
        len_taper = params["s2s_len_taper"]
        MMI_METCH_buffer = params["s2s_O_MMI_METCH_buffer"]
        METCH_taper_buffer = params["s2s_O_METCH_taper_buffer"]


    #slot etch, clkwise from top left
    x_pts = [len_MMI,
             len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
             len_MMI + len_taper,
             len_MMI + len_taper,
             len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
             len_MMI]
    y_pts = [s2s_slot_start_width/2,
             s2s_slot_start_width/2,#w_slot/2,
             w_slot/2,
             -w_slot/2,
             -s2s_slot_start_width/2,#-w_slot/2,
             -s2s_slot_start_width/2
             ]
    c.add_polygon([x_pts, y_pts], layer=SLOT_ETCH)


    # RIB Core Polygon
    xpts = [-len_in, 0, 0,
            len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
            len_MMI + len_taper,
            #len_MMI + len_taper,
            #len_MMI,
            #len_MMI,
            #len_MMI + len_taper,
            len_MMI + len_taper,
            len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
            0, 0, -len_in]
    ypts = [width_in/2, width_in/2, width_MMI/2,
            width_MMI/2,
            (w_slot/2 + w_slotWG),
            #w_slot/2,
            #w_slot/2,
            #-w_slot/2,
            #-w_slot/2,
            -(w_slot/2 + w_slotWG),
            -width_MMI/2,
            -width_MMI/2, -width_in/2, -width_in/2]
    c.add_polygon([xpts, ypts], layer=RIB)


    xpts_slabcor = [#-len_in,
                    #0,
                    #0,
                    len_MMI + MMI_METCH_buffer,
                    len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
                    len_MMI + len_taper,
                    len_MMI + len_taper,
                    len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
                    len_MMI + MMI_METCH_buffer]
                    #0,
                    #0,
                    #-len_in]
    ypts_slabcor = [#width_in / 2,
                    #width_in / 2,
                    #width_MMI / 2,
                    width_MMI / 2,
                    width_MMI / 2,
                    (w_slot / 2 + w_slotWG),
                    -(w_slot / 2 + w_slotWG),
                    -width_MMI / 2,
                    -width_MMI / 2]
                    #-width_MMI / 2,
                    #-width_in / 2,
                    #-width_in / 2]
    c.add_polygon([xpts_slabcor, ypts_slabcor], layer=SLAB_COR)

    #METCH_CLD
    sections = []
    offset_si_contact = w_slot / 2 + w_slotWG + params["w_si_contact"] / 2
    sections.append(gf.Section(width=2*params["w_si_contact"]+0.64, offset=0, layer=SLAB, name="s2s_SLAB"))
    #sections.append(gf.Section(width=params["w_si_contact"], offset=-offset_si_contact, layer=SLAB, name="s2s_SLAB"))
    x1 = gf.CrossSection(sections=sections)#, components_along_path=components_along_path)
    p1 = gf.path.straight(length=len_taper - MMI_METCH_buffer)
    _ = c << gf.path.extrude(p1, x1)
    _.movex(len_MMI + MMI_METCH_buffer)

    #OXOP
    # if "s2s_0_len_OXOP" > 0, add OXOP and doping
    if params["s2s_O_len_OXOP"] > 0:
        #OXOP extend into s2s
        coords = ([len_MMI+ 0.2 + (len_taper-params["s2s_O_len_OXOP"]), w_OXOP/2 ], [len_MMI+ 0.2+len_taper, w_OXOP/2], [len_MMI+ 0.2+len_taper, -w_OXOP/2], [len_MMI+ 0.2 + (80-params["s2s_O_len_OXOP"]), -w_OXOP/2])
        c.add_polygon(coords, layer=OXOP)

        #doping into s2s
        w_NIM = params["w_NIM"]
        w_IND = params["w_IND"]
        w_NCONT = params["w_NCONT"]
        w_si_contact = params["w_si_contact"]
        num_rows_CONTACT = params["num_rows_CONTACT"]
        gap_via_contact = params["gap_via_contact"]
        via_size_contact = params["via_size_contact"]
        gap_si_contact = params["gap_si_contact"]
        offset_si_contact = w_slot / 2 + w_slotWG + params["w_si_contact"] / 2
        w_slab = params["w_slab"]

        sections = []
        components_along_path = []

        # sections.append(gf.Section(width=2 * w_slab + w_slot + 2 * w_slotWG, offset=0, layer=SLAB, name="s2s_SLAB"))
        # x1 = gf.CrossSection(sections=sections)
        # p1 = gf.path.straight(length=L1 + L1 + L2 + L3)
        # _ = c << gf.path.extrude(p1, x1)
        # _.movex(-L1)

        s24 = sections.append(gf.Section(width=w_NIM, offset=0, layer=NIM, name="NIM"))

        offset_IND = w_slot / 2 + w_slotWG + params["gap_IND_WG"] + w_IND / 2
        s22 = sections.append(gf.Section(width=w_IND, offset=offset_IND, layer=IND, name="IND_1"))
        s23 = sections.append(gf.Section(width=w_IND, offset=-offset_IND, layer=IND, name="IND_2"))

        # w_NCONT = w_slab / 2 + offset_slab - (offset_IND + w_IND / 2) + w_RY_outside_SE
        offset_NCONT = w_slot / 2 + w_slotWG + params["gap_NCONT_WG"] + params["w_NCONT"] / 2
        s20 = sections.append(gf.Section(width=w_NCONT, offset=(offset_NCONT), layer=NCONT, name="NCONT_1"))
        s21 = sections.append(gf.Section(width=w_NCONT, offset=-(offset_NCONT), layer=NCONT, name="NCONT_2"))

        # offset_via_contact = offset_si_contact + w_si_contact / 2 - (num_rows_CONTACT - 1) * (gap_via_contact + via_size_contact[0]) + gap_si_contact  # align with slot WG slab
        # for i in range(num_rows_CONTACT):
        #     components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding=via_size_contact[1] + gap_via_contact,
        #                                                     offset=(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))
        #     components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding=via_size_contact[1] + gap_via_contact,
        #                                                     offset=-(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))

        x1 = gf.CrossSection(sections=sections)
        p1 = gf.path.straight(length=0.2+params["s2s_O_len_OXOP"])
        _ = c << gf.path.extrude(p1, x1)
        _.movex(len_MMI+0.2+len_taper-params["s2s_O_len_OXOP"]-0.2)

        # sections_dummy = []
        # sections_dummy.append(gf.Section(width=0, offset=0, layer=SLAB, name="dummy"))
        # x1 = gf.CrossSection(sections=sections_dummy, components_along_path=components_along_path)
        # p1 = gf.path.straight(length=0.2+params["s2s_O_len_OXOP"])
        # _ = c << gf.path.extrude(p1, x1)
        # _.movex(len_taper-params["s2s_O_len_OXOP"]-0.2)


    # Ports
    c.add_port("o1", center=(-len_in, 0), width=width_in, orientation=180, layer=RIB, port_type="optical")
    c.add_port("o2", center=(len_MMI + len_taper, 0), width=w_slot, orientation=0, layer=RIB, port_type="optical")

    # SLAB region
    #x_top = [len_MMI, len_MMI + len_taper, len_MMI + len_taper, len_MMI]
    x_top = [len_MMI + len_taper, len_MMI + len_taper, len_MMI + MMI_METCH_buffer+METCH_taper_buffer]

    if not extra_slab:
        SLAB_min_w = 0.3
        if Oband_variant:
            y_top = [#width_MMI/2 ,#- SLAB_min_w,
                     w_slot/2 + w_slotWG ,#- buffer_RIB_SLAB_overlay,
                     w_slot/2 + w_slotWG + w_slab,
                     w_slot/2 + (width_MMI - w_slot)/2]
            y_bot = [-y for y in y_top]

        else:
            y_top = [#(w_slot/2 + (width_MMI - w_slot)/2 ),#- SLAB_min_w),
                     (w_slot/2 + w_slotWG ),#- buffer_RIB_SLAB_overlay),
                     (w_slot/2 + w_slotWG + w_slab),
                     (w_slot/2 + (width_MMI - w_slot)/2)]
            y_bot = [-y for y in y_top]

        c.add_polygon([x_top, y_top], layer=RIB)
        c.add_polygon([x_top, y_bot], layer=RIB)
    elif extra_slab:
        OXSLAB_min_w = 0.4
        if Oband_variant:
            SLAB_min_w = 0.3
            y_top = [width_MMI/2 - SLAB_min_w, w_slot/2 + w_slotWG - buffer_RIB_SLAB_overlay,
                     w_slot/2 + w_slotWG + w_slab, w_slot/2 + (width_MMI - w_slot)/2]
        else:
            y_top = [(w_slot/2 + (width_MMI - w_slot)/2 - OXSLAB_min_w), (w_slot/2 + w_slotWG - buffer_RIB_SLAB_overlay),
                     (w_slot/2 + w_slotWG + w_slab), (w_slot/2 + (width_MMI - w_slot)/2)]
        y_bot = [-y for y in y_top]
        c.add_polygon([x_top, y_top], layer=SLAB)
        c.add_polygon([x_top, y_bot], layer=SLAB)

        offset_extra_slab = width_MMI/2 + params["s2s_gap_MMI_extra_slab"] + OXSLAB_min_w / 2
        s18 = gf.Section(width=OXSLAB_min_w, offset=offset_extra_slab, layer=SLAB)
        s19 = gf.Section(width=OXSLAB_min_w, offset=-offset_extra_slab, layer=SLAB)

        extra_slab_doping_w = 2
        extra_slab_doping_buffer_overlay = 0.1
        offset_extra_slab_doping = width_MMI/2 + params["s2s_gap_MMI_extra_slab"] + extra_slab_doping_w / 2 - extra_slab_doping_buffer_overlay
        s20 = gf.Section(width=extra_slab_doping_w, offset=offset_extra_slab_doping, layer=NIM)
        s21 = gf.Section(width=extra_slab_doping_w, offset=-offset_extra_slab_doping, layer=NIM)

        x2 = gf.CrossSection(sections=[s18, s19, s20, s21])
        p2 = gf.path.straight(length=len_taper + len_MMI)
        extra_slabs = gf.path.extrude(p2, x2)
        d_extra = c << extra_slabs
        d_extra.movex(-len_in / 2)


    return c


@gf.cell
def s2s_powertaper(params: dict) -> gf.Component:
    """

    param: taper_pwr : pwr in the taper eqaution ,1 means linear taper
    rail_step : control discretization of taper   defalut 10nm
    :return:

    example:

    c = gf.Component("Mirach_diff_MZM_GSGSG_s2s_adiabatic_sample_2025_12_12_v1")
    combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True, "s2s_type": "power"}
    _ = c << MZM_SilTerra(combined_params)
    c.show()
    """
    c = gf.Component()
    # Unpack shared parameters
    if "S2S_PWR_taper_pwr" in params:
        taper_pwr = params["S2S_PWR_taper_pwr"]
    else: taper_pwr = 0.3
    if "S2S_PWR_rail_step" in params:
        rail_step = params["S2S_PWR_rail_step"]
    else: rail_step = 0.01 #0.010

    w_slot = params["w_slot"]
    w_OXOP = params["w_OXOP"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    w_NIM = params["w_NIM"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]
    delta_to_len_taper_NIM = params["delta_to_len_taper_NIM"]
    extra_slab = params["extra_slab"]
    Oband_variant = params["Oband_variant"]

    s2s_slot_start_width = params["s2s_slot_start_width"]


    # Choose parameter names based on variant
    if Oband_variant:
        len_in = params["s2s_O_len_in"]
        width_in = params["s2s_O_w_in"]
        len_MMI = params["s2s_O_len_MMI"]
        width_MMI = params["s2s_O_w_MMI"]
        MMI_METCH_buffer=params["s2s_O_MMI_METCH_buffer"]
        METCH_taper_buffer=params["s2s_O_METCH_taper_buffer"]
        len_taper = params["s2s_O_len_taper"]
    else:
        len_in = params["s2s_len_in"]
        width_in = params["s2s_width_in"]
        len_MMI = params["s2s_len_MMI"]
        width_MMI = params["s2s_width_MMI"]
        len_taper = params["s2s_len_taper"]
        MMI_METCH_buffer = params["s2s_O_MMI_METCH_buffer"]
        METCH_taper_buffer = params["s2s_O_METCH_taper_buffer"]


    #slot etch, clkwise from top left
    x_pts = [len_MMI,
             len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
             len_MMI + len_taper,
             len_MMI + len_taper,
             len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
             len_MMI]
    y_pts = [s2s_slot_start_width/2,
             s2s_slot_start_width/2,#w_slot/2,
             w_slot/2,
             -w_slot/2,
             -s2s_slot_start_width/2,#-w_slot/2,
             -s2s_slot_start_width/2
             ]
    c.add_polygon([x_pts, y_pts], layer=SLOT_ETCH)
   
        
        
    rail_w0 = width_MMI/2 - w_slot/2
    rail_w1 = w_slotWG
        
    #rail_step = 0.01
        
    x_tap =  np.arange(0  , len_taper -  MMI_METCH_buffer - METCH_taper_buffer+  rail_step/2, rail_step)
        
    w_tap = rail_w0 + (rail_w1 - rail_w0) * (x_tap/(len_taper -  MMI_METCH_buffer - METCH_taper_buffer))**(taper_pwr)
        
    x_tap =len_MMI + MMI_METCH_buffer + METCH_taper_buffer + x_tap
    y_tap = w_slot/2 + w_tap
    y_tap_n = -1 * y_tap
        
        
    y_tap_n = y_tap_n[1:].tolist()
    y_tap = y_tap[1:].tolist()
    x_tap = x_tap[1:].tolist()
        
        
    #plt.plot(x_tap, y_tap)
        
    xpts = [-len_in, 0, 0, len_MMI + MMI_METCH_buffer + METCH_taper_buffer] 
    xpts = xpts + x_tap + x_tap[::-1] + xpts[::-1]
        
        
    ypts = [width_in/2, width_in/2, width_MMI/2,  width_MMI/2]
        
    ypts = ypts + y_tap + y_tap_n[::-1]
        
        
    ypts = ypts + [-width_MMI/2,-width_MMI/2, -width_in/2, -width_in/2]
        
    #plt.plot(xpts, ypts)
        
   
    c.add_polygon([xpts, ypts], layer=RIB)
        
    xpts_slabcor = [#-len_in,
                        #0,
                        #0,
                        len_MMI + MMI_METCH_buffer,
                        len_MMI + MMI_METCH_buffer + METCH_taper_buffer] 
    xpts_slabcor =  xpts_slabcor + x_tap + x_tap[::-1] + xpts_slabcor[::-1]
                        # len_MMI + len_taper,
                        # len_MMI + len_taper,
                        # len_MMI + MMI_METCH_buffer + METCH_taper_buffer,
                        # len_MMI + MMI_METCH_buffer]
                        # #0,
                        # #0,
                        # #-len_in]
    ypts_slabcor = [#width_in / 2,
                        #width_in / 2,
                        #width_MMI / 2,
                        width_MMI / 2,
                        width_MMI / 2]
    ypts_slabcor =  ypts_slabcor + y_tap + y_tap_n[::-1]+ [-width_MMI / 2, - width_MMI / 2]
                        # (w_slot / 2 + w_slotWG),
                        # -(w_slot / 2 + w_slotWG),
                        # -width_MMI / 2,
                        # -width_MMI / 2]
                        # #-width_MMI / 2,
                        # #-width_in / 2,
                        # #-width_in / 2]
    plt.plot(xpts_slabcor, ypts_slabcor)                
    c.add_polygon([xpts_slabcor, ypts_slabcor], layer=SLAB_COR)

    


   

    #METCH_CLD
    sections = []
    offset_si_contact = w_slot / 2 + w_slotWG + params["w_si_contact"] / 2
    sections.append(gf.Section(width=2*params["w_si_contact"]+0.64, offset=0, layer=SLAB, name="s2s_SLAB"))
    #sections.append(gf.Section(width=params["w_si_contact"], offset=-offset_si_contact, layer=SLAB, name="s2s_SLAB"))
    x1 = gf.CrossSection(sections=sections)#, components_along_path=components_along_path)
    p1 = gf.path.straight(length=len_taper - MMI_METCH_buffer)
    _ = c << gf.path.extrude(p1, x1)
    _.movex(len_MMI + MMI_METCH_buffer)


    # Ports
    c.add_port("o1", center=(-len_in, 0), width=width_in, orientation=180, layer=RIB, port_type="optical")
    c.add_port("o2", center=(len_MMI + len_taper, 0), width=w_slot, orientation=0, layer=RIB, port_type="optical")

    # SLAB region
    #x_top = [len_MMI, len_MMI + len_taper, len_MMI + len_taper, len_MMI]
    x_top = [len_MMI + len_taper, len_MMI + len_taper, len_MMI + MMI_METCH_buffer+METCH_taper_buffer, len_MMI + MMI_METCH_buffer+METCH_taper_buffer]

    if not extra_slab:
        SLAB_min_w = 0.3
        
        y_top = [#width_MMI/2 ,#- SLAB_min_w,
                     w_slot/2 + w_slotWG+w_slab ,#- buffer_RIB_SLAB_overlay,
                     -w_slot/2 - w_slotWG - w_slab,
                     -w_slot/2 - (width_MMI - w_slot)/2,
                     w_slot/2 + (width_MMI - w_slot)/2]
        

     

        c.add_polygon([x_top, y_top], layer=RIB)
        #c.add_polygon([x_top, y_bot], layer=RIB)
    elif extra_slab:
        OXSLAB_min_w = 0.4
        if Oband_variant:
            SLAB_min_w = 0.3
            y_top = [width_MMI/2 - SLAB_min_w, w_slot/2 + w_slotWG - buffer_RIB_SLAB_overlay,
                     w_slot/2 + w_slotWG + w_slab, w_slot/2 + (width_MMI - w_slot)/2]
        else:
            y_top = [(w_slot/2 + (width_MMI - w_slot)/2 - OXSLAB_min_w), (w_slot/2 + w_slotWG - buffer_RIB_SLAB_overlay),
                     (w_slot/2 + w_slotWG + w_slab), (w_slot/2 + (width_MMI - w_slot)/2)]
        y_bot = [-y for y in y_top]
        c.add_polygon([x_top, y_top], layer=SLAB)
        c.add_polygon([x_top, y_bot], layer=SLAB)

        offset_extra_slab = width_MMI/2 + params["s2s_gap_MMI_extra_slab"] + OXSLAB_min_w / 2
        s18 = gf.Section(width=OXSLAB_min_w, offset=offset_extra_slab, layer=SLAB)
        s19 = gf.Section(width=OXSLAB_min_w, offset=-offset_extra_slab, layer=SLAB)

        extra_slab_doping_w = 2
        extra_slab_doping_buffer_overlay = 0.1
        offset_extra_slab_doping = width_MMI/2 + params["s2s_gap_MMI_extra_slab"] + extra_slab_doping_w / 2 - extra_slab_doping_buffer_overlay
        s20 = gf.Section(width=extra_slab_doping_w, offset=offset_extra_slab_doping, layer=NIM)
        s21 = gf.Section(width=extra_slab_doping_w, offset=-offset_extra_slab_doping, layer=NIM)

        x2 = gf.CrossSection(sections=[s18, s19, s20, s21])
        p2 = gf.path.straight(length=len_taper + len_MMI)
        extra_slabs = gf.path.extrude(p2, x2)
        d_extra = c << extra_slabs
        d_extra.movex(-len_in / 2)

    return c


@gf.cell
def s2s_adiabatic(params: dict) -> gf.Component:
    '''

    :param params:
    :return:

    assuming Oband
    example:
    c = gf.Component("Mirach_diff_MZM_GSGSG_s2s_adiabatic_sample_2025_12_05")
combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True, "s2s_type": "adiabatic",
                   "W": 0.38,
                   "W0": 0.13,
                   "R": 0.18,
                   "S": 0.16,
                   "G": 0.16,
                   "L1": 4,
                   "L2": 4,
                   "L3": 8,
                   "B1": 2.5,
                   "B2": 0.9,
                   "B3": 2,
                   "C1": 0.5,
                   "C2": 0.13,
            }
combined_params["PS_length"] = 600
_ = c << MZM_SilTerra(combined_params)
#_.rotate(-90)
c.show()
    '''
    c = gf.Component("s2s_adiabtic")

    '''
    pseudocode: 
    match origin w s2s_from_params
    define linear sections first 
    define linear dividing lines: D1, D2, etc. : rail outer boundaries, slot inner boundaries
    create and place polygons from lines
    
    example:
    c = gf.Component("s2s_adiabatic_sample_2025_12_11_v1")
combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True,
                   "W": 0.38,
                   "T": 0.3,
                   "W0": 0.18,
                   "R": 0.2,
                   "S": 0.17,
                   "G": 0.17,
                   "L1": 4,
                   "L2": 4,
                   "L3": 4,
                   "B1": 2.25,
                   "B2": 0.87,
                   "B3": 5,
                   "C1": 0.525,
                   "C2": 0.13
            }
    
    '''
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    s2s_O_len_in = params["s2s_O_len_in"]
    offset_si_contact = w_slot / 2 + w_slotWG + params["w_si_contact"] / 2

    len_in = params["s2s_O_len_in"]
    width_in = params["s2s_O_w_in"]
    len_MMI = params["s2s_O_len_MMI"]
    width_MMI = params["s2s_O_w_MMI"]
    MMI_METCH_buffer = params["s2s_O_MMI_METCH_buffer"]
    METCH_taper_buffer = params["s2s_O_METCH_taper_buffer"]
    len_taper = params["s2s_O_len_taper"]
    w_OXOP = params["w_OXOP"]

    W = params["S2S_ADIA_W"]
    W0 = params["S2S_ADIA_W0"]
    R = params["w_slotWG"]
    S = params["w_slot"]
    T = params["S2S_ADIA_T"]
    G = params["S2S_ADIA_G"]
    L1 = params["S2S_ADIA_L1"]
    L2 = params["S2S_ADIA_L2"]
    L3 = params["S2S_ADIA_L3"]
    B1 = params["S2S_ADIA_B1"]
    B2 = params["S2S_ADIA_B2"]
    B3 = params["S2S_ADIA_B3"]
    C1 = params["S2S_ADIA_C1"]
    C2 = params["S2S_ADIA_C2"]

    x0 = np.linspace(-L1, 0, 50)
    x1 = np.linspace(0, L1, 50)
    x2 = np.linspace(L1, L1+L2, 50)
    x3 = np.linspace(L1+L2, L1+L2+L3, 500)
    x_all = np.linspace(0, L1+L2+L3, 100) #just for debugging
    A1 = (C1-W0) / np.log10(B1*L1+1)
    A2 = (R-C2)/np.log10(B2*L2+1)
    A3 = (T-R)/np.log10(B3*L3+1)
    y_W1 = C1-A1*np.log10(B1*(L1-x1)+1)
    y_W2 = C2+A2*np.log10(B2*(x2-L1)+1)
    y_W3 = T-A3*np.log10(B3*(x3-L1-L2)+1)
    y_0 = C1 - A1*np.log10(B1*(L1+x0)+1) - W0
    y_1 = C1 - A1*np.log10(B1*(x1)+1) - W0

    """divide in 4 boundary lines, left to right:"""
    #slope = rise/run =
    #f(L1 + L2) = -W/2

    L1_slope = (T/2-W/2)/(2*L1)
    y_S1 = L1_slope*(x1+L1) + W/2 #right half of top of W to T taper
    y_S0 = L1_slope * (x0+L1) + W/2  # left half of top of W to T taper

    L1_lower_slope = (W/2-T/2)/(2*L1)
    y_S1_lower = L1_lower_slope*(x1+L1) - W/2 #right half of bottom of W to T taper

    L3_slope =  ( S/2 -(-T/2))/L3
    y_S3 = L3_slope*(x3 -(L1+L2)) -T/2; #bottom of upper rail in L3

    L3_lower_slope =  ( -S/2 -(-T/2-G))/L3
    y_S3_lower = L3_lower_slope*(x3 -(L1+L2)) -T/2 - G; #top of lower rail in L3

    #D1 (from bottom left of upper input slab to top right of upper slab
    x_D1 = np.concatenate((x0, x1))
    y_D1 = np.concatenate((y_0 +y_S0 +W0, y_W1+ y_S1))
    x__ = [L1, L1+L2+L3]
    #y__ = [w_slab, w_slab]
    y__ = [offset_si_contact+params["w_si_contact"]/2, offset_si_contact+params["w_si_contact"]/2]
    x_D1 = np.concatenate((x_D1, x__))
    y_D1 = np.concatenate((y_D1, y__))
    
    #D2 (from top of input WG to top of upper rail in slotWG
    x_D2 = [-s2s_O_len_in -L1, -L1, L1, L1+L2]
    y_D2 = [W/2,            W/2,T/2,T/2]
    x_D2 = np.concatenate((x_D2, x3))
    y_D2 = np.concatenate((y_D2, y_W3 + y_S3))

    #D3 (from bottom of input WG to bottom of upper rail in slotWG
    x_D3 = [-s2s_O_len_in-L1, -L1, L1, L1+L2]
    y_D3 = [-W/2,           -W/2,-T/2,-T/2]
    x_D3 = np.concatenate((x_D3, x3))
    y_D3 = np.concatenate((y_D3, y_S3))

    #D4 (from top of lower input rail to top of lower slotWG rail
    x_D4 = x2
    y_D4 = -T/2 - G - R + y_W2
    x_D4 = np.concatenate((x_D4, x3))
    y_D4 = np.concatenate((y_D4, y_S3_lower))

    #D5 (from bottom of lower input rail to bottom of lower slotWG rail
    x_D5 = [L1,             L1+L2]
    y_D5 = [-T/2 - G - R, -T/2 - G - R]
    x_D5 = np.concatenate((x_D5, x3))
    y_D5 = np.concatenate((y_D5, y_S3_lower - R))

    #D6 (from bottom left of lower input slab to botom right of lower slab
    x_D6 = [L1, L1+L2+L3]
    y_D6 = [-(offset_si_contact+params["w_si_contact"]/2), -(offset_si_contact+params["w_si_contact"]/2)]

    #add P1
    x_p = np.concatenate((x_D1, np.flip(x_D2[2:])))
    y_p = np.concatenate((y_D1, np.flip(y_D2[2:])))
    x_p = np.concatenate((x_p, np.flip(x0)))
    y_p = np.concatenate((y_p, np.flip(y_0+y_S0)))
    P1 = gf.Polygon(list(zip(x_p, y_p)), layer=WG_LowRib)
    c.add_polygon(P1)

    # #add P2 - central FETCH_COR
    # x_p = np.concatenate((x_D2, np.flip(x_D5)))
    # y_p = np.concatenate((y_D2, np.flip(y_D5)))
    # x_p = np.concatenate((x_p, [L1, -L1, -s2s_O_len_in-L1])) #round out the bottom left
    # y_p = np.concatenate((y_p, [-T/2, -W/2, -W/2]))
    # P2 = gf.Polygon(list(zip(x_p, y_p)), layer=RIB)
    # c.add_polygon(P2)
    # # c.add_polygon([x_p, y_p], layer=RIB)

    #add P2_2 - central METCH_COR
    x_D2_truncate = x_D2[1:]
    y_D2_truncate = y_D2[1:]
    x_D2_truncate = np.concatenate(([-L1-s2s_O_len_in, -L1], x_D2_truncate))
    y_D2_truncate = np.concatenate(([W/2, W/2], y_D2_truncate))
    x_p = np.concatenate((x_D2_truncate, np.flip(x_D5)))
    y_p = np.concatenate((y_D2_truncate, np.flip(y_D5)))
    x_p = np.concatenate((x_p, [L1, -L1, -L1-s2s_O_len_in]))
    y_p = np.concatenate((y_p, [-T/2, -W/2, -W/2]))
    P2 = gf.Polygon(list(zip(x_p, y_p)), layer=WG_HM)
    c.add_polygon(P2, layer=WG_Strip) #save for slot removal

    #add P3 - slot etch
    x_D3_truncate = np.concatenate((x1, x_D3[2:]))
    y_D3_truncate = np.concatenate((y_S1_lower- y_1, y_D3[2:]))
    x_p = np.concatenate((x_D3_truncate, np.flip(x_D4)))
    y_p = np.concatenate((y_D3_truncate, np.flip(y_D4)))
    x_p = np.concatenate((x_p, np.flip(x1)))
    y_p = np.concatenate((y_p, np.flip(y_S1_lower-(G+R-C2) - y_1)))
    P3 = gf.Polygon(list(zip(x_p, y_p)), layer=WG_HM)
    c.add_polygon(P3, layer=WG_Strip)

    c_P2 = gf.Component()
    c_P2.add_polygon(P2)
    c_P3 = gf.Component()
    c_P3.add_polygon(P3)
    c_sub_result = gf.geometry.boolean(A=c_P2, B=c_P3, operation="not", layer=WG_HM)
    c << c_sub_result

    # #add P4
    # x_p = np.concatenate((x_D4, np.flip(x_D5)))
    # y_p = np.concatenate((y_D4, np.flip(y_D5)))
    # P4 = gf.Polygon(list(zip(x_p, y_p)), layer=RIB)
    # c.add_polygon(P4)

    #add P5
    x_p = np.concatenate((x_D5, np.flip(x_D6)))
    y_p = np.concatenate((y_D5, np.flip(y_D6)))
    P5 = gf.Polygon(list(zip(x_p, y_p)), layer=WG_LowRib)
    c.add_polygon(P5)


    coords = ([-L1-s2s_O_len_in, offset_si_contact+params["w_si_contact"]/2], [L1, offset_si_contact+params["w_si_contact"]/2],
              [L1, -offset_si_contact-params["w_si_contact"]/2], [-L1-s2s_O_len_in, -offset_si_contact-params["w_si_contact"]/2])
    P6 = gf.Polygon(coords, layer=WG_Strip)

    c_P5 = gf.Component()
    c_P5.add_polygon(P5)
    c_P6 = gf.Component()
    c_P6.add_polygon(P6)
    c_sub_result = gf.geometry.boolean(A=c_P6, B=c, operation="not", layer=WG_Strip)
    c << c_sub_result
    # plt.plot(x_D5, y_D5)
    # plt.show()

    # plt.plot(x2, y_W2)
    # plt.show()


    #METCH_CLD, doping, contacts, OXOP


    coords = ([L1 + 0.1, w_OXOP/2 ], [L1+L2+L3, w_OXOP/2], [L1+L2+L3, -w_OXOP/2], [L1+0.1, -w_OXOP/2])
    c.add_polygon(coords, layer=OXOP)

    w_NIM = params["w_NIM"]
    w_IND = params["w_IND"]
    w_NCONT = params["w_NCONT"]
    w_si_contact = params["w_si_contact"]
    num_rows_CONTACT = params["num_rows_CONTACT"]
    gap_via_contact = params["gap_via_contact"]
    via_size_contact = params["via_size_contact"]
    gap_si_contact = params["gap_si_contact"]
    offset_si_contact = w_slot / 2 + w_slotWG + params["w_si_contact"] / 2
    w_slab = params["w_slab"]

    sections = []
    components_along_path = []
    #
    # sections.append(gf.Section(width=2*w_slab + w_slot+2*w_slotWG, offset=0, layer=SLAB, name="s2s_SLAB"))
    # x1 = gf.CrossSection(sections=sections)
    # p1 = gf.path.straight(length=L1+L1+L2+L3)
    # _ = c << gf.path.extrude(p1, x1)
    # _.movex(-L1)



    s24 = sections.append(gf.Section(width=w_NIM, offset=0, layer=NIM, name="NIM"))

    offset_IND = w_slot/2 + w_slotWG + params["gap_IND_WG"] + w_IND/2
    s22 = sections.append(gf.Section(width=w_IND, offset=offset_IND, layer=IND, name="IND_1"))
    s23 = sections.append(gf.Section(width=w_IND, offset=-offset_IND, layer=IND, name="IND_2"))

    #w_NCONT = w_slab / 2 + offset_slab - (offset_IND + w_IND / 2) + w_RY_outside_SE
    offset_NCONT = w_slot/2 + w_slotWG + params["gap_NCONT_WG"] + params["w_NCONT"]/2
    s20 = sections.append(gf.Section(width=w_NCONT, offset=(offset_NCONT), layer=NCONT, name="NCONT_1"))
    s21 = sections.append(gf.Section(width=w_NCONT, offset=-(offset_NCONT), layer=NCONT, name="NCONT_2"))

    # offset_via_contact = offset_si_contact + w_si_contact / 2 - (num_rows_CONTACT - 1) * (gap_via_contact + via_size_contact[0]) + gap_si_contact  # align with slot WG slab
    # for i in range(num_rows_CONTACT):
    #         components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding=via_size_contact[1] + gap_via_contact,
    #                                                         offset=(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))
    #         components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding=via_size_contact[1] + gap_via_contact,
    #                                                         offset=-(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))

    x1 = gf.CrossSection(sections=sections)
    p1 = gf.path.straight(length=0.1+L2+L3)
    _ = c << gf.path.extrude(p1, x1)
    _.movex(L1-0.1)

    sections_dummy = []
    sections_dummy.append(gf.Section(width=0, offset=0, layer=SLAB, name="s2s_SLAB"))
    x1 = gf.CrossSection(sections=sections_dummy, components_along_path=components_along_path)
    p1 = gf.path.straight(length=0.1 + L2 + L3 + 1.25)
    _ = c << gf.path.extrude(p1, x1)
    _.movex(L1 - 0.1)

    # Ports
    c.add_port("o1", center=(-L1-s2s_O_len_in, 0), width=width_in, orientation=180, layer=RIB, port_type="optical")
    c.add_port("o2", center=(L1+L2+L3, 0), width=w_slot, orientation=0, layer=RIB, port_type="optical")


    return c

@gf.cell
def PS_slotWG_SilTerra(params: dict, position="") -> gf.Component:
    """
        Unified phase shifter slot waveguide with all parameter variations.

        Parameters:
        - w_slot: slot width
        - w_OXOP: oxide opening width
        - PS_length: phase shifter length
        - gap_IND_WG: gap between inductor and waveguide
        - w_slotWG: slot waveguide width (defaults to global constant if None)
        - gap_NCONT_WG: gap between N-contact and waveguide (defaults to global constant if None)
        - gap_si_contact: gap between silicon contact and waveguide (defaults to global constant if None)
        - buffer_RIB_SLAB_overlay: slab overlay buffer (defaults to global constant if None)
        """
    Oband_variant = params["Oband_variant"]
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    if "PS_sipho_length" not in params:
        PS_length = params["PS_length"]
    else:
        PS_length = params["PS_sipho_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]
    #len_taper = params["len_taper"]
    #w_RY_outside_SE = params["w_RY_outside_SE"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_FETCH_CLD = params["w_FETCH_CLD"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    via_size_contact = params["via_size_contact"]
    silicide_width = params["silicide_width"]
    gap_silicide = params["gap_silicide"]
    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size_1 = params["via_size_1"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]
    w_MT2 = params["w_MT2"]
    via_size_top = params["via_size_top"]
    gap_via_top = params["gap_via_top"]

    electrical = params["electrical"]
    num_rows_V2 = params["num_rows_V2"]
    num_rows_V1 = params["num_rows_V1"]
    gap_via_contact = params["gap_via_contact"]
    num_rows_CONTACT = params["num_rows_CONTACT"]

    # Choose parameter names based on variant
    if Oband_variant:
        len_in = params["s2s_O_len_in"]
        width_in = params["s2s_O_w_in"]
        len_MMI = params["s2s_O_len_MMI"]
        width_MMI = params["s2s_O_w_MMI"]
        len_taper = params["s2s_O_len_taper"]
    else:
        len_in = params["s2s_len_in"]
        width_in = params["s2s_width_in"]
        len_MMI = params["s2s_len_MMI"]
        width_MMI = params["s2s_width_MMI"]
        len_taper = params["s2s_len_taper"]

    c = gf.Component("PS_slotWG_from_params")

    sections_short_short = []
    sections_short_sh = []
    sections_short = []
    sections = []
    sections_doping_extend = []
    sections_electrical_extend = []
    sections_extended = []
    sections_extended_full = []
    sections_extended_extra = []
    components_along_path = []
    components_along_path_electrical_extend = []

    s0 = sections.append(gf.Section(width = w_slot, offset=0, layer=SLOT_ETCH))

    offset_slotWG = (w_slotWG + w_slot) / 2

    s1 = sections.append(gf.Section(width = w_slot+2*w_slotWG, offset=0, layer=RIB))
    # s1 = sections.append(gf.Section(width=w_slotWG, offset=offset_slotWG, layer=RIB, name="slotWG_1"))
    # s2 = sections.append(gf.Section(width=w_slotWG, offset=-offset_slotWG, layer=RIB, name="slotWG_2"))

    # s1 = sections.append(gf.Section(width=w_slotWG + offset_slotWG/2, offset=offset_slotWG*.75, layer=SLAB_COR, name="slotWG_1"))
    # s2 = sections.append(gf.Section(width=w_slotWG + offset_slotWG/2, offset=-offset_slotWG*.75, layer=SLAB_COR, name="slotWG_2"))

    s1 = sections.append(gf.Section(width=2*w_slotWG+w_slot, offset=0, layer=SLAB_COR, name="slotWG_1"))

    offset_slab = (w_slot + w_slab) / 2 + w_slotWG
    # s3 = sections.append(gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=offset_slab, layer=SLAB, name="slab_1"))
    # s4 = sections.append(gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=-offset_slab, layer=SLAB, name="slab_2"))
    s4 = sections.append(gf.Section(width=(2*w_slab + w_slot+2*w_slotWG), offset=0, layer=SLAB, name="slab_2"))

    # offset_si_contact = w_slot / 2 + w_slotWG + gap_si_contact + w_si_contact / 2
    # s5 = sections.append(gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact_1"))
    # s6 = sections.append(gf.Section(width=w_si_contact, offset=-offset_si_contact, layer=RIB, name="sl_contact_2"))

    offset_si_contact = w_slot / 2 + w_slotWG + w_si_contact / 2
    s5 = sections_extended.append(gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact_1"))
    s6 = sections_extended.append(gf.Section(width=w_si_contact, offset=-offset_si_contact, layer=RIB, name="sl_contact_2"))

    s24 = sections_doping_extend.append(gf.Section(width=w_NIM, offset=0, layer=NIM, name="NIM"))

    offset_IND = w_slot/2 + w_slotWG + gap_IND_WG + w_IND/2
    s22 = sections_doping_extend.append(gf.Section(width=w_IND, offset=offset_IND, layer=IND, name="IND_1"))
    s23 = sections_doping_extend.append(gf.Section(width=w_IND, offset=-offset_IND, layer=IND, name="IND_2"))

    #w_NCONT = w_slab / 2 + offset_slab - (offset_IND + w_IND / 2) + w_RY_outside_SE
    offset_NCONT = w_slot/2 + w_slotWG + gap_NCONT_WG + w_NCONT/2
    s20 = sections_doping_extend.append(gf.Section(width=w_NCONT, offset=(offset_NCONT), layer=NCONT, name="NCONT_1"))
    s21 = sections_doping_extend.append(gf.Section(width=w_NCONT, offset=-(offset_NCONT), layer=NCONT, name="NCONT_2"))

    s_FETCH_CLD = sections_extended_full.append(gf.Section(width=w_FETCH_CLD, offset=0, layer=RIB_ETCH, name="FETCH_CLD"))
    #s_NFE_CLD = sections_extended_full.append(gf.Section(width=w_FETCH_CLD, offset=0, layer=NOP, name="NOP"))

    #s_SWG_dummy_block_PS = sections_extended.append(gf.Section(width=2*(w_slot + w_slotWG + w_slab + w_si_contact ), offset = 0, layer=SWG_DUMMY_BLOCK, name="SWG_DUMMY_BLOCK"))

    #s_WGCNCT_PS_connected = sections_extended_extra.append(gf.Section(width=w_slotWG*2 + w_slab*2 + 2, offset=0, layer=WGCNCT, name="WGCNCT"))

    s_SWG_dummy_block_S2S = sections_extended_full.append(gf.Section(width=w_slotWG*2 + w_slab*2 + 2, offset=0, layer=SWG_DUMMY_BLOCK, name="SWG_DUMMY_BLOCK"))


    if electrical:
        #working from outside to inside

        #via locations are set automatically from metal widths and other slot wg spacings
        offset_MT1 = w_OXOP / 2 + min_gap_OXOP_MT + w_MT1 / 2  # offset_si_contact + w_MT1/2 - (num_rows_CONTACT-1)*(gap_via_contact)
        if params["MT1_from_PS"]: #if regular GSGSG, add M1 lines
            sections_electrical_extend.append(gf.Section(width=w_MT1, offset=offset_MT1, layer=MT1, name="MT1_1"))
            sections_electrical_extend.append(gf.Section(width=w_MT1, offset=-offset_MT1, layer=MT1, name="MT1_1"))

        # if w_MT2 == -1:
        offset_MT2 = w_OXOP / 2 + min_gap_OXOP_MT + w_MT2 / 2  #= params["PS_inner_gap_width"]/1.94 + 0.5*params["PS_center_gnd_width"] - w_MT2/2

        # # MT1 and V1AM from MT1 to MT2

        s7 = sections_electrical_extend.append(gf.Section(width=0, layer=MT1, name="MT1_1")) #CrossSection requires a dummy section for whatever reason

        offset_via_1 = offset_si_contact + w_si_contact/2 - (num_rows_V1)*(via_size_top+gap_via_top) + gap_via_1
        for i in range(num_rows_V1):
            if position=="right":
                components_along_path_electrical_extend.append(ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top+gap_via_top, enclosure=min_inc_via_1,
                                                                offset=(offset_via_1 + i * (via_size_top + gap_via_top))))
            elif position=="left":
                components_along_path_electrical_extend.append(ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top+gap_via_top, enclosure=min_inc_via_1,
                                                                offset=-(offset_via_1 + i * (via_size_top + gap_via_top))))
            else:
                pass
                # components_along_path_electrical_extend.append(
                #     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top, enclosure=min_inc_via_1,
                #                        offset=(offset_via_1 + i * (via_size_top + gap_via_top))))
                # components_along_path_electrical_extend.append(
                #     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top, enclosure=min_inc_via_1,
                #                        offset=-(offset_via_1 + i * (via_size_top + gap_via_top))))

        # vias from doped rib slab to MT1
        offset_via_contact = offset_si_contact + w_si_contact/2 - (num_rows_CONTACT-1)*(gap_via_contact+via_size_contact[0]) +gap_si_contact #align with slot WG slab
        for i in range(num_rows_CONTACT):
            if position=="right":
                components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding= via_size_contact[1] + gap_via_contact,
                                                                offset=(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))
            elif position=="left":
                components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding= via_size_contact[1] + gap_via_contact,
                                                                offset=-(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))
            else:
                components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding=via_size_contact[1] + gap_via_contact,
                                                                offset=(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))
                components_along_path.append(ComponentAlongPath(component=gf.c.via(size=via_size_contact, layer=CONTACT), spacing=via_size_contact[0] + gap_via_contact, padding=via_size_contact[1] + gap_via_contact,
                                                                offset=-(offset_via_contact + i * (via_size_contact[1] + gap_via_contact))))
        #silicide open, match Contact width and location
        #silicide_width = (num_rows_CONTACT+2.2) * (gap_via_contact+via_size_contact[0]) + sb_overplot
        offset_silicide = offset_via_contact + gap_silicide
        s_silicide_contact_top = sections_short_sh.append(gf.Section(width=silicide_width, offset=(offset_silicide), layer=SB, name="SB_PS"))
        s_silicide_contact_bot = sections_short_sh.append(gf.Section(width=silicide_width, offset=-(offset_silicide), layer=SB, name="SB_PS"))

        s_silicide_contact_top = sections_extended.append(gf.Section(width=silicide_width, offset=(offset_silicide), layer=SLAB_COR, name="SB_PS"))
        s_silicide_contact_bot = sections_extended.append(gf.Section(width=silicide_width, offset=-(offset_silicide), layer=SLAB_COR, name="SB_PS"))
        s_silicide_contact_top = sections_extended.append(gf.Section(width=silicide_width, offset=(offset_silicide), layer=SLAB, name="SB_PS"))
        s_silicide_contact_bot = sections_extended.append(gf.Section(width=silicide_width, offset=-(offset_silicide), layer=SLAB, name="SB_PS"))
        s_silicide_contact_top = sections_extended.append(gf.Section(width=silicide_width, offset=(offset_silicide), layer=RIB, name="SB_PS"))
        s_silicide_contact_bot = sections_extended.append(gf.Section(width=silicide_width, offset=-(offset_silicide), layer=RIB, name="SB_PS"))


        s_LT = sections.append(gf.Section(width=w_OXOP, layer=OXOP, name="oxide open"))
        #s152 = sections_short.append(gf.Section(width=16, layer=NOP, name="nitride_open"))

        #s_M1_LT_etch_stop = sections_short_sh.append(gf.Section(width=w_OXOP+4, layer=MT1, name="M1_LT_trench_stop "))

        # w_M1_etch_stop = w_OXOP/2 -(w_slotWG/2+offset_slotWG)
        # s_M1_etch_stop_top = sections.append(gf.Section(width=w_M1_etch_stop, offset=(w_M1_etch_stop/2 + offset_slotWG + w_slotWG/2), layer=MT1, name="M1 etch stop"))
        # s_M1_etch_stop_bot = sections.append(gf.Section(width=w_M1_etch_stop, offset=-(w_M1_etch_stop / 2 + offset_slotWG + w_slotWG / 2), layer=MT1, name="M1 etch stop"))
        #
        # s_M1_etch_stop_slot = sections.append(gf.Section(width=w_slot, layer=MT1, name="M1 etch stop"))


    #length of PS
    x1 = gf.CrossSection(sections=sections)
    p1 = gf.path.straight(length=PS_length)
    PS = gf.path.extrude(p1, x1)

    #electrical=False
    if electrical:
        x1_2 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path_electrical_extend)
        p1_2 = gf.path.straight(length=PS_length)# + 2*params["extension_electrical"])
        PS_e = gf.path.extrude(p1_2, x1_2)

    x1_1 = gf.CrossSection(sections=sections_doping_extend)
    p1_1 = gf.path.straight(length=PS_length+0.4)
    PS_dop = gf.path.extrude(p1_1, x1_1)
    #length of PS - VG1 min gap
    vg1_gap = 15
    x4 = gf.CrossSection(sections=sections_short)
    p4 = gf.path.straight(length=PS_length - 2 * vg1_gap)
    PS_short = gf.path.extrude(p4, x4)

    m1_overplot_lt = 2
    x6 = gf.CrossSection(sections=sections_short_sh, components_along_path=components_along_path)
    if params["s2s_type"] == "adiabatic":
        p6 = gf.path.straight(length=PS_length - .4 + 2*(params["S2S_ADIA_L2"] + params["S2S_ADIA_L3"]) + 0.1)
        PS_short_sh = gf.path.extrude(p6, x6)
        _ = c << PS_short_sh
        _.movex(-(params["S2S_ADIA_L2"] + params["S2S_ADIA_L3"] + 0.1) + 0.25)  # vg1_gap + vg1_overplot_lt - m1_overplot_lt)
    else:
        p6 = gf.path.straight(length=PS_length - .4)
        PS_short_sh = gf.path.extrude(p6, x6)
        _ = c << PS_short_sh
        _.movex(0.2)  # vg1_gap + vg1_overplot_lt - m1_overplot_lt)


    vg1_overplot_lt = 3
    x5 = gf.CrossSection(sections=sections_short_short)
    p5 = gf.path.straight(length=PS_length - 2*5)
    PS_short_short = gf.path.extrude(p5, x5)


    #length of PS + SB overplot RWG (0.5)
    if params["s2s_type"] == "adiabatic":
        x2 = gf.CrossSection(sections=sections_extended)
        p2 = gf.path.straight(length=PS_length + 2*(params["S2S_ADIA_L2"]+params["S2S_ADIA_L3"]))#+ 1)
        PS_extended = gf.path.extrude(p2, x2)
        _ = c << PS_extended
        _.movex(-(params["S2S_ADIA_L2"]+params["S2S_ADIA_L3"]))  # -0.5)
    else:
        x2 = gf.CrossSection(sections=sections_extended)
        p2 = gf.path.straight(length=PS_length)#+ 1)
        PS_extended = gf.path.extrude(p2, x2)
        _ = c << PS_extended
        _.movex(0)  # -0.5)


    #length of PS + length of S2S inside slab
    x3 = gf.CrossSection(sections=sections_extended_extra)
    p3 = gf.path.straight(length=PS_length + 2*(len_taper))
    PS_extended_extra = gf.path.extrude(p3, x3)

    #length of PS + length of S2S
    x8 = gf.CrossSection(sections=sections_extended_full)
    if Oband_variant:
        if params["s2s_type"] == "adiabatic":
            p8 = gf.path.straight(length=PS_length + 2*params["s2s_O_len_in"] + 2*(2*params["S2S_ADIA_L1"]+ params["S2S_ADIA_L2"]+params["S2S_ADIA_L3"]) )
        else: p8 = gf.path.straight(length=PS_length + 2*(len_taper) + 2*len_MMI + 2*params["s2s_O_len_in"])
    else: p8 = gf.path.straight(length=PS_length + 2*(len_taper) + 2*len_MMI + 2*params["s2s_len_in"])
    PS_extended_full = gf.path.extrude(p8, x8)

    _ = c << PS
    if electrical:
        _ = c << PS_e
        #_.movex(-params["extension_electrical"])
    _ = c << PS_dop
    _.movex(-0.2)
    _ = c << PS_short
    _.movex(vg1_gap)
    _ = c << PS_short_short
    _.movex(5)
    _ = c << PS_extended_extra
    _.movex(-(len_taper))
    _ = c << PS_extended_full
    if params["s2s_type"] == "adiabatic":
        _.movex(-((2*params["S2S_ADIA_L1"]+ params["S2S_ADIA_L2"]+params["S2S_ADIA_L3"]) + params["s2s_O_len_in"]))
    else:
        _.movex(-(len_taper + len_MMI + params["s2s_O_len_in"]))

    # Clear existing ports and add custom ports
    c.ports.clear()

    c.add_port(
        name="o1",
        center=(0, 0),
        width=w_slot,
        orientation=180,
        layer=RIB,
        port_type="optical",
    )

    c.add_port(
        name="o2",
        center=(PS_length, 0),
        width=w_slot,
        orientation=0,
        layer=RIB,
        port_type="optical",
    )

    return c

@gf.cell
def PS_slotWG_from_params(params: dict, Oband_variant) -> gf.Component:
    """
        AMF version
        Unified phase shifter slot waveguide with all parameter variations.

        Parameters:
        - w_slot: slot width
        - w_OXOP: oxide opening width
        - PS_length: phase shifter length
        - gap_IND_WG: gap between inductor and waveguide
        - w_slotWG: slot waveguide width (defaults to global constant if None)
        - gap_NCONT_WG: gap between N-contact and waveguide (defaults to global constant if None)
        - gap_si_contact: gap between silicon contact and waveguide (defaults to global constant if None)
        - buffer_RIB_SLAB_overlay: slab overlay buffer (defaults to global constant if None)
        """

    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]

    # Choose parameter names based on variant
    if Oband_variant:
        len_in = params["s2s_O_len_in"]
        width_in = params["s2s_O_w_in"]
        len_MMI = params["s2s_O_len_MMI"]
        width_MMI = params["s2s_O_w_MMI"]
        len_taper = params["s2s_O_len_taper"]
    else:
        len_in = params["s2s_len_in"]
        width_in = params["s2s_width_in"]
        len_MMI = params["s2s_len_MMI"]
        width_MMI = params["s2s_width_MMI"]
        len_taper = params["s2s_len_taper"]

    #print("PS_slotWG buffer_RIB_SLAB_overlay: " + str(buffer_RIB_SLAB_overlay))
    offset_slotWG = (w_slotWG + w_slot)/2
    s1 = gf.Section(width=w_slotWG, offset=offset_slotWG, layer=RIB, name="slotWG_1")
    s2 = gf.Section(width=w_slotWG, offset=-offset_slotWG, layer=RIB, name="slotWG_2")

    offset_slab = (w_slot + w_slab)/2 + w_slotWG
    s3 = gf.Section(width=(w_slab+ buffer_RIB_SLAB_overlay*2), offset=offset_slab, layer=SLAB, name="slab_1")
    s4 = gf.Section(width=(w_slab+ buffer_RIB_SLAB_overlay*2), offset=-offset_slab, layer=SLAB, name="slab_2")

    offset_si_contact = w_slot/2 + w_slotWG + gap_si_contact + w_si_contact/2
    s5 = gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact_1")
    s6 = gf.Section(width=w_si_contact, offset=-offset_si_contact, layer=RIB, name="sl_contact_2")

    offset_MT1 = w_OXOP/2 + w_MT1/2 + min_gap_OXOP_MT
    s7 = gf.Section(width=w_MT1, offset=offset_MT1, layer=MT1, name="MT1_1")
    s8 = gf.Section(width=w_MT1, offset=-offset_MT1, layer=MT1, name="MT1_2")

    offset_via_1 = w_OXOP/2 + min_gap_OXOP_MT + min_inc_via_1 + via_size/2
    s9v = ComponentAlongPath(component=gf.c.via(size=(3,3), layer=VIA1), spacing=via_size + gap_via_1, padding=via_size, enclosure=min_inc_via_1, offset=offset_via_1)
    s10v = ComponentAlongPath(component=gf.c.via(size=(3,3), layer=VIA1), spacing=via_size + gap_via_1, padding=via_size, enclosure=min_inc_via_1, offset=-offset_via_1)

    offset_via_2 = w_OXOP/2 + min_gap_OXOP_MT + min_inc_via_1 + via_size + min_exc_of_via_2 + via_size/2
    s11v = ComponentAlongPath(component=gf.c.via(size=(3,3), layer=VIA2), spacing=min_exc_of_via_2 + via_size, padding=via_size, offset=offset_via_2)
    s12v = ComponentAlongPath(component=gf.c.via(size=(3,3), layer=VIA2), spacing=min_exc_of_via_2 + via_size, padding=via_size, offset=-offset_via_2)

    offset_NCONT = w_slot/2 + w_slotWG + gap_NCONT_WG + w_NCONT/2
    s13 = gf.Section(width=w_NCONT, offset=offset_NCONT, layer=NCONT, name="NCONT_1")
    s14 = gf.Section(width=w_NCONT, offset=-offset_NCONT, layer=NCONT, name="NCONT_2")

    offset_IND = w_slot/2 + w_slotWG + gap_IND_WG + w_IND/2
    s15 = gf.Section(width=w_IND, offset=offset_IND, layer=IND, name="IND_1")
    s16 = gf.Section(width=w_IND, offset=-offset_IND, layer=IND, name="IND_2")

    s17 = gf.Section(width=w_NIM, offset=0, layer=NIM, name="NIM")

    s151 = gf.Section(width=w_OXOP, layer=OXOP, name="oxide_open")

    x1 = gf.CrossSection(
        sections=(s1, s2, s3, s4, s5, s6, s7, s8, s13, s14, s15, s16, s17, s151),
        components_along_path=[s9v, s10v, s11v, s12v],
        )

    p1 = gf.path.straight(length=PS_length)
    PS = gf.path.extrude(p1, x1)

    # Clear existing ports and add custom ports
    PS.ports.clear()

    PS.add_port(
        name = "o1",
        center = (0, 0),
        width = w_slot,
        orientation = 180,
        layer = RIB,
        port_type = "optical",
        )

    PS.add_port(
        name = "o2",
        center = (PS_length, 0),
        width = w_slot,
        orientation = 0,
        layer = RIB,
        port_type = "optical",
        )

    return PS


@gf.cell
def PS_connected_from_params(params: dict, position="") -> gf.Component:
    """
    changed to Tower version
    #relies on PS_slotWG_from_params, s2s_from_params
    # Oband_variant used only in DOE5, s2s_type used only in DOE8 (as of 8/15/2025)
    # s2s_type can be "FNG", "oxide", "extra_slab"
    #position paramter is used in PS_slotWG in case of shared central ground ("left" or "right") - if "left" then only left electical contacts will be placed, vice versa for "right"

    """
    if "s2s_type" in params:
        s2s_type = params["s2s_type"]
    else:
        s2s_type = "MMI"

    c = gf.Component()

    # Create PS waveguide
    PS = PS_slotWG_SilTerra(params, position=position)
    ref_PS = c << PS

    # s2s_w_OXOP: float = None
    # if s2s_type == "oxide":
    #     params["w_OXOP"] = 0
    # elif s2s_type == "FNG":
    #     pass
    # elif s2s_type == "extra_slab":
    #     params["extra_slab"] = True  # extra slab bool set in function params for DOE1-5 - can also be set through s2s_type also
    # else:
    #     pass  # default fallback

    # Create input taper and connect
    if s2s_type == "adiabatic":
        s2s_in = s2s_adiabatic(params)

    elif s2s_type == "power":
        s2s_in = s2s_powertaper(params)
    else:
        s2s_in = s2s_from_params(params)
    ref_s2s_in = c << s2s_in
    ref_s2s_in.connect("o2", ref_PS.ports["o1"])

    # Create output taper and connect
    #s2s_out = s2s_from_params(params, extra_slab=extra_slab, Oband_variant=Oband_variant)
    ref_s2s_out = c << s2s_in
    ref_s2s_out.connect("o2", ref_PS.ports["o2"])

    # Add electrical port at center
    electrical_start = (0, 0)
    # if params["s2s_type"] == "adiabatic":
    #     electrical_start = (-(params["S2S_ADIA_L2"]+params["S2S_ADIA_L3"]), 0)
    c.add_port(
        name="e_MT2",
        center=electrical_start,
        width=1,
        orientation=180,
        layer=MT2,
        port_type="electrical",
    )
    offset_MT1 = params["w_OXOP"] / 2 + params["min_gap_OXOP_MT"] + params["w_MT1"] / 2
    c.add_port(name="e1", center=(params["ps_connected_total_length"], offset_MT1), width=0.5, orientation=0, layer=MT1,
                port_type="electrical")
    c.add_port(name="e2", center=(params["ps_connected_total_length"], -offset_MT1), width=0.5, orientation=0, layer=MT1,
               port_type="electrical")

    # Add optical ports
    c.add_port("o1", port=ref_s2s_in.ports["o1"])
    c.add_port("o2", port=ref_s2s_out.ports["o1"])

    return c


@gf.cell
def PS_connected_s2s_only(params: dict, extra_slab: bool = False, Oband_variant: bool = False, s2s_type="FNG") -> gf.Component:
    '''
    PS connected without PS
    used for TCDoe s2s cascade test structure
    :param params:
    :param extra_slab:
    :param Oband_variant:
    :param s2s_type:
    :return:
    '''
    # s2s_type can be "FNG", "oxide", "extra_slab"
    c = gf.Component()

    s2s_w_OXOP: float = None
    if s2s_type == "oxide":
        params = {**params, "w_OXOP": 0}
    elif s2s_type == "FNG":
        pass
    elif s2s_type == "extra_slab":
        extra_slab = True  # extra slab bool set in function params for DOE1-5 - can also be set through s2s_type also
    else:
        pass  # default fallback

    # Create input taper and connect
    s2s_in = s2s_from_params(params, extra_slab=extra_slab, Oband_variant=Oband_variant)
    ref_s2s_in = c << s2s_in
    #ref_s2s_in.connect("o2", ref_PS.ports["o1"])

    # Create output taper and connect
    s2s_out = s2s_from_params(params, extra_slab=extra_slab, Oband_variant=Oband_variant)
    ref_s2s_out = c << s2s_out
    #ref_s2s_out.connect("o2", ref_PS.ports["o2"])

    ref_s2s_in.connect("o2", ref_s2s_out["o2"])

    sections_extended_full = []
    if Oband_variant:
        len_MMI = params["s2s_O_len_MMI"]
        len_taper = params["s2s_O_len_taper"]
        len_in = params["s2s_O_len_in"]
    else:
        len_MMI = params["s2s_len_MMI"]
        len_taper = params["s2s_len_taper"]
        len_in = params["s2s_len_in"]
    w_FETCH_CLD = params["w_FETCH_CLD"]

    #add nitride removal, full etch
    s_FETCH_CLD = sections_extended_full.append(gf.Section(width=w_FETCH_CLD, offset=0, layer=RIB_ETCH, name="FETCH_CLD"))
    s_NFE_CLD = sections_extended_full.append(gf.Section(width=w_FETCH_CLD, offset=0, layer=NOP, name="NOP"))
    x8 = gf.CrossSection(sections=sections_extended_full)
    p8 = gf.path.straight(length= 2*len_taper + 2*len_MMI + 2*len_in)
    PS_extended_full = gf.path.extrude(p8, x8)
    _ = c << PS_extended_full
    _.movex(-(len_in))

    # Add optical ports
    c.add_port("o1", port=ref_s2s_in.ports["o1"])
    c.add_port("o2", port=ref_s2s_out.ports["o1"])

    return c



@gf.cell
def MZM_SilTerra(params: dict) -> gf.Component:
    """
    new version so that parameters can easily be read from CSV and placed in batches
    :param params:
    :param trans_length:
    :param s2s_type: can be "FNG" (default), "oxide", "extra_slab"
    :param extra_slab:
    :param Oband_variant:


    The below are overrides and specifications for GSGSG_MT2
    :param electrode_params

    :param taper_type: s2s overrides, 1 or 2
    :param sig_trace: ps overrides, "narrow" "medium" "wide"
    :param ps_config: ps parameter overrides: "default"(default), "narrow_custom" "medium_custom" "wide_custom":
    :param termination:
    if termination = 0, the terminating electrode pads will match the default pads for a symmetric electrode structure
            if termination = 35, 50, 65, output pad geometry will be changed to the specified resistance. Also, heaters between the termination pads will be added.
                also, these additional parameters must be specified in parameter dictionary: "htr_width_x", "htr_length", "htr_connect_length",
                                                                                            "sc_length", "pad_t_length",  "trans_length_to_term"
    :param gnds_shorted: bool, if True ground shorting structure will be added
    :param gsgsg_variant: almost obsolete, only used to specify SGS_MT2_DC in DOE8
    :param config: standard (default), batch, compact - used only in SGS_MT2_DC
    :return:

    example:
    c = gf.Component("Mirach_diff_MZM_GSGSG_taper_sample_2025_12_05")
    combined_params = {**differential_electrode_params, **balun_sipho_params,  "PS_trans_length": 250}
    combined_params["PS_length"] = 1000
    _ = c << MZM_SilTerra(combined_params)
    #_.rotate(-90)
    c.show()

    c = gf.Component("Mirach_diff_MZM_GSGSG_s2s_adiabatic_sample_2025_12_05")
combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True, "s2s_type": "adiabatic",
                   "W": 0.4,
                   "W0": 0.13,
                   "R": 0.18,
                   "S": 0.16,
                   "G": 0.16,
                   "L1": 4,
                   "L2": 4,
                   "L3": 8,
                   "B1": 2.5,
                   "B2": 0.9,
                   "B3": 2,
                   "C1": 0.5,
                   "C2": 0.13,
            }
combined_params["PS_length"] = 600
_ = c << MZM_SilTerra(combined_params)
#_.rotate(-90)
c.show()
    """
    #imports for legacy sub-functions
    trans_length = params["trans_length"]
    taper_type = params["taper_type"]
    sig_trace = params["sig_trace"]
    termination = params["termination"]
    extra_slab = params["extra_slab"]
    config = params["config"]
    gsgsg_variant = params["gsgsg_variant"]
    Oband_variant = params["Oband_variant"]
    s2s_type  = params["s2s_type"]
    ps_config = params["ps_config"]
    gnds_shorted = params["gnds_shorted"]


    c = gf.Component()

    # Create both arms of PS_connected
    PS1 = PS_connected_from_params(params)
    ref_PS_1 = c << PS1

    PS2 = PS_connected_from_params(params)
    ref_PS_2 = c << PS2

    # Choose GSGSG implementation
    PS_length = params["PS_length"]
    if gsgsg_variant == "DC":
        # DOE8 specific DC variant
        GSGSG = SGS_MT2_DC(PS_length, trans_length, taper_type, sig_trace, config=config, params=params)
    # elif gsgsg_variant == "bond":
    #     GSGSG = GSGSG_MT2_symmetric(PS_length, trans_length, taper_type, sig_trace, electrode_params, termination=termination, ps_config=ps_config, gnds_shorted=True)
    else:
        # Standard GSGSG variant
        GSGSG = GSGSG_MT2(PS_length, trans_length, taper_type, sig_trace, params=params, termination=termination, gnds_shorted = gnds_shorted, ps_config = ps_config)

    ref_SGS = c << GSGSG

    # Connect electrical ports
    ref_PS_1.connect("e_MT2", ref_SGS.ports["e_up"])
    ref_PS_2.connect("e_MT2", ref_SGS.ports["e_low"])

    # Add optical ports
    c.add_port("o1", port=ref_PS_1["o1"])
    c.add_port("o3", port=ref_PS_1["o2"])
    c.add_port("o2", port=ref_PS_2["o1"])
    c.add_port("o4", port=ref_PS_2["o2"])

    return c




# @gf.cell
# def MZM_GSG( #unused for SilTerra
#         params: dict,
#         #trans_length: float,
#         taper_type: int,  # 1, 2, or 3
#         #termination: int,  # 35, 50, or 65 (ohms)
#         sig_trace: str,  # test_GSG_1 - test_GSG_4
#         extra_slab: bool = False,
#         Oband_variant: bool = False,
#         s2s_type: str = "FNG"  # s2s_type can be "FNG", "oxide", "extra_slab"
# ) -> gf.Component:
#
#     c = gf.Component("MZM_GSG")
#     # PS = PS_connected_EES21(w_slot, w_OXOP, PS_length, gap_IND_WG)
#     PS = PS_connected_from_params(params=params, s2s_type=s2s_type)
#     ref_PS_1 = c << PS
#
#     # GSG = GSG_MT2_EES21(PS_length, trans_length, sig_trace, taper_type)
#     GSG = GSG_MT2(params, taper_type, sig_trace)
#     ref_GSG = c << GSG
#
#     ref_PS_1.connect("e_MT2", ref_GSG.ports["e_up"])
#
#     c.add_port("o1", port=ref_PS_1["o1"])
#     c.add_port("o3", port=ref_PS_1["o2"])
#
#     return c



# @gf.cell
# def MZM_GSG_balun( #unused for SilTerra
#         params: dict,
#         # trans_length: float,
#         # taper_type: int,  # 1 or 2
#         # sig_trace: str,  # "narrow", "medium", or "wide"
#         # termination: int = 0,  # 35, 50, or 65 (ohms) or 0 to match input
#         # extra_slab: bool = False,
#         # config: str = "standard",  # "standard", "batch", "compact"
#         # gsgsg_variant: str = "default",  # or "S21", "DC", "sym"
#         # Oband_variant: bool = False,
#         # s2s_type: str = "FNG",  # s2s_type can be "FNG", "oxide", "extra_slab"
#         # ps_config: str = "unchanged", #"unchanged", "narrow_custom", "medium_custom", "wide_custom"
#         # gnds_shorted: bool = False
#
# ) -> gf.Component:
#     """
#     ex:  mzm_1_ref = c << MZM_GSG_balun(mzm_1_params)
#     new version so that parameters can easily be read from CSV and placed in batches
#
#     ***functions called:***
#     PS_connected_from_parmams
#         s2s_from_params
#         PS_slotWG_SilTerra  - includes si contacts, off-center vias
#     GSG_MTX              - includes central vias, metal extension, vias in extension
#
#     ************************
#     :param params:
#     :param trans_length:
#     :param s2s_type: can be "FNG" (default), "oxide", "extra_slab"
#     :param extra_slab:
#     :param Oband_variant:
#     :param taper_type: s2s overrides, 1 or 2
#     :param sig_trace: ps overrides, "narrow" "medium" "wide"
#     :param ps_config: ps parameter overrides: "unchanged"(default), "narrow_custom" "medium_custom" "wide_custom":
#     :param termination:
#     if termination = 0, the terminating electrode pads will match the default pads for a symmetric electrode structure
#             if termination = 35, 50, 65, output pad geometry will be changed to the specified resistance. Also, heaters between the termination pads will be added.
#                 also, these additional parameters must be specified in parameter dictionary: "htr_width_x", "htr_length", "htr_connect_length",
#                                                                                             "sc_length", "pad_t_length",  "trans_length_to_term"
#     :param gnds_shorted: bool, if True ground shorting structure will be added
#     :param gsgsg_variant: almost obsolete, only used to specify SGS_MT2_DC in DOE8
#     :param config: standard (default), batch, compact - used only in SGS_MT2_DC
#     :return:
#
#     """
#
#     #imports for legacy sub-functions
#     trans_length = params["trans_length"]
#     taper_type = params["taper_type"]
#     sig_trace = params["sig_trace"]
#     termination = params["termination"]
#     extra_slab = params["extra_slab"]
#     # config = params["config"]
#     # gsgsg_variant = params["gsgsg_variant"]
#     # Oband_variant = params["Oband_variant"]
#     # s2s_type  = params["s2s_type"]
#     # ps_config = params["ps_config"]
#     # gnds_shorted = params["gnds_shorted"]
#
#
#     num_rows_V1 = params["num_rows_V1"]
#
#     c = gf.Component("MZM_GSG_balun")
#
#     c = gf.Component()
#
#     # Create both arms of PS_connected
#     PS1 = PS_connected_from_params(params, position="right")
#     ref_PS_1 = c << PS1
#
#     PS2 = PS_connected_from_params(params, position="left")
#     ref_PS_2 = c << PS2
#
#     # Standard GSG variant
#     GSG = GSG_MTX(params, taper_type, sig_trace, termination=termination, layer_MTX = MT2)
#     ref_GSG = c << GSG
#
#     # Connect electrical ports
#     ref_PS_1.connect("e_MT2", ref_GSG.ports["e_up"])
#     ref_PS_2.connect("e_MT2", ref_GSG.ports["e_low"])
#
#     # Add optical ports
#     c.add_port("o1", port=ref_PS_1["o1"])
#     c.add_port("o3", port=ref_PS_1["o2"])
#     c.add_port("o2", port=ref_PS_2["o1"])
#     c.add_port("o4", port=ref_PS_2["o2"])
#
#     return c




@gf.cell
def MMI_2X2_bend_s():
    c = gf.Component()

    ref_MMI = c << MMI_2X2
    #Si_WG_pitch=10
    #print("2x2 Si_WG_pitch:")
    #print(Si_WG_pitch)
    bend_s = gf.components.bend_s(size=[20, Si_WG_pitch - 0.667], npoints=40, cross_section=rib_Oband) #Si_WG_pitch/2-0.667

    bend_1 = c << bend_s
    bend_1.connect("o1", ref_MMI.ports["o1"])

    bend_2 = c << bend_s
    bend_2.mirror_y()
    bend_2.connect("o1", ref_MMI.ports["o2"])

    bend_3 = c << bend_s
    bend_3.connect("o1", ref_MMI.ports["o3"])

    bend_4 = c << bend_s
    bend_4.mirror_y()
    bend_4.connect("o1", ref_MMI.ports["o4"])

    c.add_port("o1", port=bend_1.ports["o2"])
    c.add_port("o2", port=bend_2.ports["o2"])
    c.add_port("o3", port=bend_3.ports["o2"])
    c.add_port("o4", port=bend_4.ports["o2"])

    return c


@gf.cell
def MMI_1X2_bend_s():
    c = gf.Component()

    ref_MMI = c << MMI_1X2

    bend_s = gf.components.bend_s(size=[20, Si_WG_pitch / 2 - 0.9], npoints=40, cross_section=rib_Oband)
    # bend_1 = c << bend_s
    # bend_1.connect("o1", ref_MMI.ports["o1"])

    # bend_2 = c << bend_s
    # bend_2.mirror_y()
    # bend_2.connect("o1", ref_MMI.ports["o2"])

    bend_3 = c << bend_s
    bend_3.connect("o1", ref_MMI.ports["o2"])

    bend_4 = c << bend_s
    bend_4.mirror_y()
    bend_4.connect("o1", ref_MMI.ports["o3"])

    c.add_port("o1", port=ref_MMI.ports["o1"])
    c.add_port("o3", port=bend_3.ports["o2"])
    c.add_port("o4", port=bend_4.ports["o2"])

    return c

@gf.cell 
def MMI_2X2_bend_s_O():
    c = gf.Component()

    ref_MMI = c << MMI_2X2_O
    
    bend_s = gf.components.bend_s(size=[20, Si_WG_pitch/2 -0.75], npoints=40, cross_section=rib_Oband)

    bend_1 = c << bend_s
    bend_1.connect("o1", ref_MMI.ports["o1"])
    
    bend_2 = c << bend_s
    bend_2.mirror_y()
    bend_2.connect("o1", ref_MMI.ports["o2"])
    
    bend_3 = c << bend_s
    bend_3.connect("o1", ref_MMI.ports["o3"])
    
    bend_4 = c << bend_s
    bend_4.mirror_y()
    bend_4.connect("o1", ref_MMI.ports["o4"])   

    c.add_port("o1", port=bend_1.ports["o2"])
    c.add_port("o2", port=bend_2.ports["o2"])
    c.add_port("o3", port=bend_3.ports["o2"])
    c.add_port("o4", port=bend_4.ports["o2"])   
    
    return c

@gf.cell
def MMI_1X2_bend_s_MZI_ER_config():
    c = gf.Component()

    ref_MMI = c << MMI_1X2

    bend_s = gf.components.bend_s(size=[20, Si_WG_pitch / 2 - 0.667 - 0.233], npoints=40, cross_section=rib_Oband)

    bend_1 = c << bend_s
    bend_1.connect("o1", ref_MMI.ports["o2"])

    bend_2 = c << bend_s
    bend_2.mirror_y()
    bend_2.connect("o1", ref_MMI.ports["o3"])

    straight = c << WG_straight(9, width=0.5)
    straight.connect("o1", ref_MMI.ports["o1"])

    c.add_port("o2", port=bend_1.ports["o2"])
    c.add_port("o1", port=bend_2.ports["o2"])
    c.add_port("o3", port=straight.ports["o2"])

    return c

@gf.cell
def MZI_ER_cell(type:str, combined_params:dict):

    DC_pad_size_x = combined_params["DC_pad_size_x"]
    DC_pad_size_y = combined_params["DC_pad_size_y"]

    c = gf.Component()

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place IOs
    EC_start_position = 0
    EC_length = 370
    EC_N = 2
    EC_1 = c << EC_array(EC_N, EC_pitch)
    EC_1.rotate(270).move((EC_length + 100, EC_start_position))

    # place DC pads
    DC = c << DC_pads(DC_pad_size_x, DC_pad_size_y)
    DC.mirror((0, 1)).move((325, -700))  # was 300, -700

    # place MMIs and waveguides
    match type:
        case '2x2':
            myMMI_i = c << MMI_2X2_bend_s()
            myMMI_o = c << MMI_2X2_bend_s()
        case '1x2':
            myMMI_i = c << MMI_1X2_bend_s_MZI_ER_config()
            myMMI_i.mirror((0, 1)).move((90, 0))
            myMMI_o = c << MMI_1X2_bend_s_MZI_ER_config()
        case '2x2wSWG':
            myMMI_i = c << MMI_2X2_bend_s()
            myMMI_o = c << MMI_2X2_bend_s()
            myWGc = ocdrWG(combined_params, slotW=0.2, wgOff=1, wgOff1=1.38,
                           wgWin=0.7, wgWfin=0.365, wgtL=6, xSlab='n',
                           slW=4.3, oxW=3, oxOff=2, slWgL=10,
                           dop1='n', dop2='n', dopOff=1, gapIND=1,
                           ribShape=[], ribPitch=0, ribDist=1.5, rep=1)
            myWG1 = c << myWGc
            myWG2 = c << myWGc
            myWG1.rotate(0).move((160, -450))
            myWG2.rotate(0).move((160, -350))
        case '1x2wSWG':
            myMMI_i = c << MMI_1X2_bend_s_MZI_ER_config()
            myMMI_i.mirror((0, 1)).move((90, 0))
            myMMI_o = c << MMI_1X2_bend_s_MZI_ER_config()
            myWGc = ocdrWG(combined_params, slotW=0.2, wgOff=1, wgOff1=1.38,
                           wgWin=0.7, wgWfin=0.365, wgtL=6, xSlab='n',
                           slW=4.3, oxW=3, oxOff=2, slWgL=10,
                           dop1='n', dop2='n', dopOff=1, gapIND=1,
                           ribShape=[], ribPitch=0, ribDist=1.5, rep=1)
            myWG1 = c << myWGc
            myWG2 = c << myWGc
            myWG1.rotate(0).move((160, -450))
            myWG2.rotate(0).move((160, -350))
        case _:
            return 0

    # place TPMs
    TPM_act = c << TO_PS
    TPM_act.rotate(180).move((500, -450))
    TPM_idl = c << TO_PS
    TPM_idl.rotate(180).move((500, -350))

    # place trombone for imbalance
    trs = c << trombone(75)
    trs.move((400, -545))

    myMMI_i.rotate(0).move((400, -250))
    myMMI_o.rotate(0).move((160, -550))

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    # optical
    routes = []
    # input to MMI
    routes.append(gf.routing.get_bundle_from_steps(
        EC_1.ports["o1"], myMMI_i.ports["o3"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        steps=[{'dx': 40}, {'dy': 10}]
    )[0])
    # MMI to TPM
    if 'wSWG' in type:
        routes.append(gf.routing.get_bundle_from_steps(
            myMMI_i.ports["o2"], myWG1.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dx': -240 if '2x2' in type else -280}, {'dy': 10}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            myWG1.ports["o1"], TPM_act.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            # steps=[{'dx':-240},{'dy':10}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            myMMI_i.ports["o1"], myWG2.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dx': -230 if '2x2' in type else -270}, {'dy': 10}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            myWG2.ports["o1"], TPM_idl.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            # steps=[{'dx':-240},{'dy':10}]
        )[0])
    else:
        routes.append(gf.routing.get_bundle_from_steps(
            myMMI_i.ports["o2"], TPM_act.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dx': -250}, {'dy': 10}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            myMMI_i.ports["o1"], TPM_idl.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dx': -240}, {'dy': 10}]
        )[0])
    # TPM to trombone
    routes.append(gf.routing.get_bundle_from_steps(
        TPM_act.ports["o1"], trs.ports["o2"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        steps=[{'dx': 20}, {'dy': 10}]
    )[0])
    routes.append(gf.routing.get_bundle_from_steps(
        TPM_idl.ports["o1"], trs.ports["o4"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        steps=[{'dx': 30}, {'dy': 10}]
    )[0])
    # trombone to mmi
    routes.append(gf.routing.get_bundle_from_steps(
        trs.ports["o1"], myMMI_o.ports["o3" if '2x2' in type else 'o2'],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        # steps=[{'dx':20},{'dy':10}]
    )[0])
    routes.append(gf.routing.get_bundle_from_steps(
        trs.ports["o3"], myMMI_o.ports["o4" if '2x2' in type else "o1"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        # steps=[{'dx':40},{'dy':10}]
    )[0])
    # mmi to output
    routes.append(gf.routing.get_bundle_from_steps(
        myMMI_o.ports["o2" if '2x2' in type else 'o3'], EC_1.ports["o2"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        steps=[{'dx': -30}, {'dy': 350}, {'dx': 380}]
    )[0])

    for i in routes:
        c.add(i.references)

    # electrical
    routes.append(gf.routing.get_bundle_from_steps(
        DC.ports["e1"], TPM_act.ports["e1"],
        width=20, layer=MT2,
        cross_section=gf.cross_section.metal1,
        separation=30,
        steps=[{'dy': 70}, {'dx': 350}]
    )[0])
    routes.append(gf.routing.get_bundle_from_steps(
        DC.ports["e2"], TPM_act.ports["e2"],
        width=20, layer=MT2,
        cross_section=gf.cross_section.metal1,
        separation=30,
        steps=[{'dy': 70}, {'dx': 350}]
    )[0])
    for i in routes:
        c.add(i.references)

    ### termination of floated electrical ports of TO phase shifters
    term = c << gf.components.pad(size=(20, 3), layer=MT2, port_inclusion=0, port_orientation=90)
    term.connect("e2", getattr(TPM_idl, f"ports")["e1"])
    term = c << gf.components.pad(size=(20, 3), layer=MT2, port_inclusion=0, port_orientation=90)
    term.connect("e2", getattr(TPM_idl, f"ports")["e2"])

    # show die
    # c.show()
    return c


@gf.cell
def WG_straight(length, width=0.41):
    c = gf.Component("WG_straight")
    _ = c << gf.components.straight(length=length, width=width, layer=RIB)
    c.add_port(name="o1", center=[0, 0], width=width, orientation=180, layer=RIB)
    c.add_port(name="o2", center=[length, 0], width=width, orientation=0, layer=RIB)
    return c


@gf.cell
def WG_pair_delta_L(delta_L, width=0.41, length=2):
    c = gf.Component("WG_pair_delta_L")

    _1_1 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _1_1.mirror_y()
    _1_2 = c << gf.components.straight(length=length, width=width, cross_section=rib_Oband)
    _1_2.connect('o1', _1_1.ports["o2"])
    _1_3 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _1_3.connect('o1', _1_2.ports["o2"])
    _1_4 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _1_4.connect('o1', _1_3.ports["o2"])
    _1_5 = c << gf.components.straight(length=length, width=width, cross_section=rib_Oband)
    _1_5.connect('o1', _1_4.ports["o2"])
    _1_6 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _1_6.connect('o2', _1_5.ports["o2"])

    _2_1 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _2_1.movey(Si_WG_pitch)
    _2_2 = c << gf.components.straight(length=length + delta_L / 2, width=width, cross_section=rib_Oband)
    _2_2.connect('o1', _2_1.ports["o2"])
    _2_3 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _2_3.connect('o2', _2_2.ports["o2"])
    _2_4 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _2_4.connect('o2', _2_3.ports["o1"])
    _2_5 = c << gf.components.straight(length=length + delta_L / 2, width=width, cross_section=rib_Oband)
    _2_5.connect('o1', _2_4.ports["o1"])
    _2_6 = c << gf.components.bend_euler(radius=20, angle=90, npoints=40, width=width, cross_section=rib_Oband)
    _2_6.connect('o1', _2_5.ports["o2"])

    c.add_port("o1", port=_2_1.ports["o1"])
    c.add_port("o2", port=_1_1.ports["o1"])
    c.add_port("o3", port=_2_6.ports["o2"])
    c.add_port("o4", port=_1_6.ports["o1"])

    return c

    
# DOE5 function only
@gf.cell 
def dam(dam_w, dam_len):
    c = gf.Component()
    
    xpts = [0, dam_w, dam_w, 0]
    ypts = [0, 0, dam_wall_w, dam_wall_w]
    c.add_polygon([xpts, ypts], layer=DAM)
    
    xpts = [0, dam_w, dam_w, 0]
    ypts = [dam_len-dam_wall_w, dam_len-dam_wall_w, dam_len, dam_len]
    c.add_polygon([xpts, ypts], layer=DAM)    
    
    xpts = [0, dam_wall_w, dam_wall_w, 0]
    ypts = [0, 0, dam_len, dam_len]
    c.add_polygon([xpts, ypts], layer=DAM)    
    
    xpts = [dam_w-dam_wall_w, dam_w, dam_w, dam_w-dam_wall_w]
    ypts = [0, 0, dam_len, dam_len]
    c.add_polygon([xpts, ypts], layer=DAM)     
    
    return c


# DOE5 function only
@gf.cell 
def dam_array_DOE():
    c = gf.Component()
    dam_N, dam_pitch_w, dam_pitch_len
    
    dam1_w = 1000 
    dam1_len = 1500 
    dam2_w = 1000 
    dam2_len = 2000
    dam3_w = 1800 
    dam3_len = 1700 
    dam4_w = 2000
    dam4_len = 2000 
    
    for j in range(2):
        for i in range(dam_N):
            ref_dam1 = c << dam(dam1_w, dam1_len)
            ref_dam1.move((i*dam_pitch_w + 1500, (10.-4*j)*dam_pitch_len))
            ref_dam2 = c << dam(dam2_w, dam2_len)
            ref_dam2.move(((i)*dam_pitch_w + 1500, (9.-4*j)*dam_pitch_len)) 
            ref_dam3 = c << dam(dam3_w, dam3_len)
            ref_dam3.move(((i)*dam_pitch_w + 1500, (8.-4*j)*dam_pitch_len))         
            ref_dam4 = c << dam(dam4_w, dam4_len)
            ref_dam4.move(((i)*dam_pitch_w + 1500, (7.-4*j)*dam_pitch_len))   

    for i in range(dam_N):
        ref_dam1 = c << dam(dam1_w, dam1_len)
        ref_dam1.move((i*dam_pitch_w + 1500, (10.-4*2)*dam_pitch_len))            
        ref_dam2 = c << dam(dam2_w, dam2_len)
        ref_dam2.move(((i)*dam_pitch_w + 1500, (9.-4*2)*dam_pitch_len))
            
    return c
    
# DOE5 function only
# dam_N = 3
@gf.cell 
def dam_array_10mm():
    c = gf.Component()
    dam_N, dam_pitch_w, dam_pitch_len
    
    dam1_w = 1000 
    dam1_len = 1500 
    dam2_w = 1000 
    dam2_len = 2000
    dam3_w = 1800 
    dam3_len = 1700 
    
    for j in range(1):
        for i in range(dam_N):
            ref_dam1 = c << dam(dam1_w, dam1_len)
            ref_dam1.move((i*dam_pitch_w + 1500, (2.5 -4*j)*dam_pitch_len))
            ref_dam2 = c << dam(dam2_w, dam2_len)
            ref_dam2.move(((i)*dam_pitch_w + 1500, (1.5 -4*j)*dam_pitch_len)) 
            ref_dam3 = c << dam(dam3_w, dam3_len)
            ref_dam3.move(((i)*dam_pitch_w + 1500, (0.5 -4*j)*dam_pitch_len))
            # ref_dam4 = c << dam(dam4_w, dam4_len)
            # ref_dam4.move(((i)*dam_pitch_w + 1500, (7.-4*j)*dam_pitch_len))
   
    return c


@gf.cell
def dam_array_autodispensing_150mm_wafer():

    @gf.cell
    def die():
        c = gf.Component()
        _ = c << Frame()
        _ = c << dam_array_DOE()
        return c

    c = gf.Component("layout_150mm")

    pitch_x = 25000
    pitch_y = 32000

    for j in range(4): #3 by 4 dies
        for i in range(3):
            _ = c << die()
            _.move((i * pitch_x, j * pitch_y))
            if j == 1 or j == 2:
                _ = c << die()
                _.move(((-1) * pitch_x, j * pitch_y))
                _ = c << die()
                _.move(((3) * pitch_x, j * pitch_y))

    return c


@gf.cell
def dtrDOE_cell():
    c = gf.Component()
    doe = [[10, 100, 0, 0], [20, 100, 0, 150], [30, 100, 0, 300],
           [10, 200, 140, 0], [20, 200, 140, 300], [30, 200, 140, 600],
           [10, 310, 400, 0], [20, 310, 400, 500], [30, 310, 400, 1000]]
    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place Dams
    for i, cell in enumerate(doe):
        p = gf.Path([(0, 0), (cell[1], 0), (cell[1], cell[1]), (0, cell[1]), (0, 0), (cell[0], 0)])
        pp = c << gf.path.extrude(p, layer=DTR, width=cell[0])
        pp.move((cell[2], cell[3]))

    # show die
    # c.show()
    return c

@gf.cell
def Nano3_litho_marks():
    c = gf.Component()

    distance_cross = 100
    distance_square = 80
    cross_top = c << gf.components.cross(length=80, width=5, layer=RIB)
    cross_top.move((-distance_cross / 2 - 10, distance_cross * 2))
    cross_bottom = c << gf.components.cross(length=80, width=5, layer=RIB)
    cross_bottom.move((-distance_cross / 2 - 10, -distance_cross))
    square_small = c << gf.components.rectangle(size=(20, 20), layer=RIB, centered=True)
    square_small.move((-distance_square, 0))
    square_med = c << gf.components.rectangle(size=(30, 30), layer=RIB, centered=True)
    square_large = c << gf.components.rectangle(size=(40, 40), layer=RIB, centered=True)
    square_large.move((distance_square, 0))

    cross_MT2_top = c << gf.components.cross(length=80, width=5, layer=MT2)
    cross_MT2_top.move((distance_cross / 2 + 10, distance_cross + distance_cross))
    cross_MT2_bottom = c << gf.components.cross(length=80, width=5, layer=MT2)
    cross_MT2_bottom.move((distance_cross / 2 + 10, -distance_cross * 2 + distance_cross))
    square_MT2_small = c << gf.components.rectangle(size=(20, 20), layer=MT2, centered=True)
    square_MT2_small.move((-distance_square, 0 + distance_cross))
    square_MT2_med = c << gf.components.rectangle(size=(30, 30), layer=MT2, centered=True)
    square_MT2_med.move((0, 0 + distance_cross))
    square_MT2_large = c << gf.components.rectangle(size=(40, 40), layer=MT2, centered=True)
    square_MT2_large.move((distance_square, 0 + distance_cross))

    dummy_line_0 = c << gf.components.rectangle(size=(200, 0.2), layer=RIB, centered=True)
    dummy_line_0.move((0, distance_cross / 2))
    dummy_line_1 = c << gf.components.rectangle(size=(200, 0.2), layer=RIB, centered=True)
    dummy_line_1.move((0, -distance_cross / 2))
    dummy_line_2 = c << gf.components.rectangle(size=(200, 0.2), layer=RIB, centered=True)
    dummy_line_2.move((0, distance_cross * 3 / 2))
    dummy_line_3 = c << gf.components.rectangle(size=(0.2, distance_cross * 2), layer=RIB, centered=True)
    dummy_line_3.move((-distance_square / 2, distance_cross / 2))
    dummy_line_4 = c << gf.components.rectangle(size=(0.2, distance_cross * 2), layer=RIB, centered=True)
    dummy_line_4.move((distance_square / 2, distance_cross / 2))
    dummy_line_5 = c << gf.components.rectangle(size=(0.2, distance_cross), layer=RIB, centered=True)
    dummy_line_5.move((0, distance_cross * 2))
    dummy_line_6 = c << gf.components.rectangle(size=(0.2, distance_cross), layer=RIB, centered=True)
    dummy_line_6.move((0, -distance_cross))

    return c




#@gf.cell
def wsclDOEcell(params):
    '''

    :param params:
    :return:     #returns c, ports_list; input MMI is o1, o2; output MMI is o3, 4


    example:
    params={**wscl2_params,**electrode_wscl2_params, **differential_electrode_params, **balun_sipho_params, "MT1_from_PS": True}
    c = gf.Component()
    _, wsclCellports = wsclDOEcell(params)
    _.show()
    '''
    psL = params['PS_length']
    tgtImbalance = 75
    trombL = 67.72
    DC_pad_size_x = params["DC_pad_size_x"]
    DC_pad_size_y = params["DC_pad_size_y"]

    ports_list = []
    c = gf.Component()

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place MZIs
    # mzm=MZM_GSGSG_DOE6(w_slot=slotW, w_OXOP=oxopW, PS_length=psL, gap_IND_WG=indWgG,
    #	 trans_length=trnL, taper_type=tt, sig_trace=sig_trace)

    #params = {**params, "w_slot": slotW, "w_OXOP": oxopW, "PS_length": psL, "gap_IND_WG": indWgG}
    #mzm = MZM_GSGSG_from_params(params, trnL, tt, sig_trace=sig_trace, electrode_params=electrode_DOE6_params, gsgsg_variant="sym")
    mzm = MZM_SilTerra(params)
    device = c << mzm

    ss = device.bbox
    
    device.rotate(0).movex(-ss[0,0])

    ### GC array
    params["wsclDOEcell_placeGC"] = 1
    N_GC = 6
    pitch_GC = 127
    
    GC_ = GC_TE
    
    GC_a =[]
    
    
    if params["wsclDOEcell_placeGC"]:
        
        N_arr = 1/2 * np.arange(-N_GC+ 1, N_GC + 1,2)
        
        for jj in range(len(N_arr)):
            
            GC_a.append( c << GC_TE)
            GC_a[-1].rotate(90)
            bb_ = GC_a[-1].bbox
            
          # GC_a[-1].movex(-bb_[0,0] - GC_a[-1].xsize/2 + N_arr[jj] * pitch_GC + device.xsize/2 +100 ).movey(-bb_[1,1] + ss[0,1] - 330)
            GC_a[-1].movex(-bb_[0,0] - GC_a[-1].xsize/2 + N_arr[jj] * pitch_GC + 688.667 ).movey(-bb_[1,1] + ss[0,1] - 330)
          
            #c.add_port ("GC_" + str(jj) , port = GC["o1"])
        
        
        
               
        # GC = c << GC_TE
        # bb_ = GC.bbox
        # GC.move( + GC.xsize/2)
        
    
    # place TPMs (with thermal isolation)
    TO_gap = 20
    TPM_act = c << TO_PS
    TPM_idl = c << TO_PS
    TPM_act.mirror_x(x0 = 0)#.mirror_y(y0 = 0)
    bb_ = TPM_act.bbox
    TPM_act.movex(-bb_[0,0] +162.5).movey(-bb_[1,1] - device.ysize/2 -210)
    TPM_act.movey(128)
    _1 = c << gf.components.rectangle(size=(20, 1.5*20 ), layer=MT2, centered=False, port_type='electrical', port_orientations=(180, 90, 0, -90))
    _1.connect('e4', TPM_act["e3"])
    _2 = c << gf.components.rectangle(size=(20, 1.5*20 ), layer=MT2, centered=False, port_type='electrical', port_orientations=(180, 90, 0, -90))
    _2.connect('e4', TPM_act["e2"])
    _3 = c << gf.components.rectangle(size=(_2['e1'].x - _1['e3'].x, 20 ), layer=MT2, centered=False, port_type='electrical', port_orientations=(180, 90, 0, -90))
    _3.connect('e1', _1["e3"])
    _3.movey(20/4)   
    
    
    TPM_idl.mirror_y(y0 = 0)#.mirror_x(x0 = 0)
    bb_ = TPM_idl.bbox
    TPM_idl.movex(-bb_[0,0] + TPM_idl.xsize + 39.1).movey(-bb_[1,1] - device.ysize/2 -100 )
    TPM_idl.movey(-110)
    
    
    # place MMIs
    # input MMIs
    mmi_i = c << MMI_2X2_bend_s()
    #mmi_i.connect('o1',GC_a[1]["o1"])
    #mmi_i.rotate(90)
    bb_=mmi_i.bbox
    mmi_i.movex(-bb_[0,0] + TPM_idl.xsize + 183.3).movey(-bb_[1,1] -155 - device.ysize/2 )
    # output MMIs
    mmi_o = c << MMI_2X2_bend_s()
    #mmi_i.connect('o1',GC_a[1]["o1"])
    #mmi_i.rotate(90)
    bb_=mmi_o.bbox
    mmi_o.rotate(90)
    # mmi_o.move(((device.xsize + psL)/2, -500))
    mmi_o.move(((device.xsize) -220, -500))

    # place DC pads
    DCpad = c << DC_pads(DC_pad_size_x, DC_pad_size_y, DC_pad_gap=10)
    bb_ = DCpad.bbox
    # DCpad.movex(-bb_[0,0]).movey(-bb_[0,1]).rotate(180).movex(device.xsize/2-450).movey(470)
    DCpad.movex(-bb_[0,0]).movey(-bb_[0,1]).rotate(180).movex(195).movey(470)
    
    
    
    # TPM_idl.rotate(180).move((device.xsize / 2 + 60, -450))
    # TPM_act.rotate(180).move((device.xsize / 2 + 60, -550))

    # place trombones (to tune MZI imbalance)
    trs = c << trombone(trombL)
    # trs.rotate(0).move(((device.xsize + psL) / 2 + 150, -155))
    trs.rotate(0).move((90, -421.05))
    

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    
    # electrical

    # DC pads to TPMs
    mySteps = [{'y': device.ysize/2+25 },{'x': -20.0},{'y':-device.ysize/2 -50}, {'x': _1["e2"].x-100}]#, {'dx': -450}, {'dy': -50}, {'dx': -400}, {'dy': -750}, {'dx': 220}, {'dy': -40}]
    #mySteps = [{'y': -20}, {'x': -20}, {'y': -550}]        
    routes = gf.routing.get_bundle_from_steps_electrical (
        DCpad.ports["e1"], _1["e2"],
        cross_section=metal1_metal2(params),
        separation=20,
        steps=mySteps,
    )
    for i in routes:
        c.add(i.references)

    mySteps = [{'y': device.ysize/2 + 10 +45},{'x': -50},{'y':TPM_act["e1"].y - 120}, {'x': TPM_act["e1"].x}]#, {'dx': -450}, {'dy': -50}, {'dx': -400}, {'dy': -750}, {'dx': 220}, {'dy': -40}]
               
    routes = gf.routing.get_bundle_from_steps_electrical (
        DCpad.ports["e2"], TPM_act["e1"],
        cross_section=metal1_metal2(params),
        separation=20,
        steps=mySteps,
    )
    for i in routes:
        c.add(i.references)

    # optical
    # GC to MMI
    routes, lengths = [], []
    ports_list.append(mmi_i.ports["o1"])
    ports_list.append(mmi_i.ports["o2"])
    #ports_list.append(mmi_o.ports["o1"])
    #ports_list.append(mmi_o.ports["o2"])
    ports_list[0].name = "o1_in"
    ports_list[1].name = "o2_in"
    #ports_list[2].name = "o3_out"
    #ports_list[3].name = "o4_out"

    if params["wsclDOEcell_placeGC"]:
        # input GC to MMI and GC loop
        routes.append(gf.routing.get_bundle_from_steps(
            GC_a[0].ports["o1"], GC_a[1].ports["o1"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dy': 30}, {'dx': 0}])[0])
        # if params["PS_length"] > 900:
        #     routes.append(gf.routing.get_bundle_from_steps(
        #         GC_a[2].ports["o1"], mmi_i.ports["o4"],
        #         width=0.38, cross_section=rib_Oband,
        #         separation=5,
        #         steps=[{'dy': 15}])[0])
        # else:
        #     routes.append(gf.routing.get_bundle_from_steps(
        #         GC_a[2].ports["o1"], mmi_i.ports["o4"],
        #         width=0.38, cross_section=rib_Oband,
        #         separation=5,
        #         steps=[{'dy': 15},{'dx': 70}])[0])    
        # routes.append(gf.routing.get_bundle_from_steps(
        #     GC_a[3].ports["o1"], mmi_i.ports["o3"],
        #     width=0.38, cross_section=rib_Oband,
        #     separation=5,
        #     steps=[{'dy': 25}])[0])
        routes.append(gf.routing.get_bundle_from_steps(
            GC_a[2].ports["o1"], mmi_i.ports["o4"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dy': 40},{'dx': 130-3.667}])[0])
        routes.append(gf.routing.get_bundle_from_steps(
            GC_a[3].ports["o1"], mmi_i.ports["o3"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dy': 20},{'dx': 30}])[0])
        # MMI to TPMs
        routes.append(gf.routing.get_bundle_from_steps(
            mmi_i.ports["o2"], TPM_act.ports["o1"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dx': -80}])[0])
        lengths.append(routes[-1].length)
        routes.append(gf.routing.get_bundle_from_steps(
            mmi_i.ports["o1"], TPM_idl.ports["o2"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dx': -25}])[0])
        lengths.append(routes[-1].length)
         # TPMs to trombone
        routes.append(gf.routing.get_bundle_from_steps(
            TPM_act.ports["o2"],trs.ports["o2"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            # steps=[{'dy': -20},{'dx': -100}]
            )[0])
        lengths.append(routes[-1].length)
        routes.append(gf.routing.get_bundle_from_steps(
            TPM_idl.ports["o1"],trs.ports["o4"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            # steps=[{'dy': 20},{'dx': -100}]
            )[0])
        lengths.append(routes[-1].length)       
        # trombone to MZI
        routes.append(gf.routing.get_bundle_from_steps(
            trs.ports["o3"],device.ports["o1"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dx': -130}])[0])
        lengths.append(routes[-1].length)
        routes.append(gf.routing.get_bundle_from_steps(
            trs.ports["o1"],device.ports["o2"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dx': -120}])[0])
        lengths.append(routes[-1].length)        
          # MZI to MMI out
        routes.append(gf.routing.get_bundle_from_steps(
            device.ports["o4"],mmi_o.ports["o3"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dx': 40}])[0])
        lengths.append(routes[-1].length)
        routes.append(gf.routing.get_bundle_from_steps(
            device.ports["o3"],mmi_o.ports["o4"],
            width=0.38, cross_section=rib_Oband,
            separation=5,
            steps=[{'dx': 60}])[0])
        lengths.append(routes[-1].length)
            #  MMI out to GCs
        if params["PS_length"] > 740:
            routes.append(gf.routing.get_bundle_from_steps(
                mmi_o.ports["o1"], GC_a[5].ports["o1"],
                width=0.38, cross_section=rib_Oband,
                separation=5,
                steps=[{'dy': -40}, {'dx': -30}])[0])
            routes.append(gf.routing.get_bundle_from_steps(
                mmi_o.ports["o2"], GC_a[4].ports["o1"],
                width=0.38, cross_section=rib_Oband,
                separation=5,
                steps=[{'dy': -20}, {'dx': -70}])[0])
        else:

            routes.append(gf.routing.get_bundle_from_steps(
                mmi_o.ports["o1"],GC_a[5].ports["o1"],
                width=0.38, cross_section=rib_Oband,
                separation=5,
                steps=[{'dy': -20},{'dx': -30}])[0])
            routes.append(gf.routing.get_bundle_from_steps(
                mmi_o.ports["o2"],GC_a[4].ports["o1"],
                width=0.38, cross_section=rib_Oband,
                separation=5,
                steps=[{'dy': -40}])[0])
        
    for i in routes:
        c.add(i.references)

    # check imbalance

    myDeltaL = lengths[0] + lengths[2] + lengths[4] + lengths[6] - (lengths[1] + lengths[3] + lengths[5] + lengths[7])
    if myDeltaL + trombL != tgtImbalance:
        print("Warning: wsclDOEcell MZI imbalance %f different from target %f" % (myDeltaL + trombL, tgtImbalance))

    '''



    mySteps = [{'dy': -100}, {'dx': -450}, {'dy': -50}, {'dx': -400}, {'dy': -750}, {'dx': 220}, {'dy': -40},
               {'dx': 100}, {'dy': -40}, {'dx': 100}, {'dy': -40}, {'dx': 100}, {'dy': -150}] if psL == 500 \
        else [{'dy': -100},{'dx': -1120}, {'dy': -900}, {'dx': 350}, {'dy': -40}, {'dx': 220}, {'dy': -170}] if psL == 1500 \
        else [{'dy': -100},{'dx': -600}, {'dy': -50}, {'dx': -240}, {'dy': -50}, {'dx': -psL/4}, {'dy': -700}, {'dx': psL/4}, {'dy': -40}, {'dx': 120}, {'dy': -40},
        {'dx': 150}, {'dy': -40}, {'dx': 200}, {'dy': -40}, {'dx': 100}, {'dy': -150}]
    routes = gf.routing.get_bundle_from_steps(
        DCpad.ports["e1"], TPM_act.ports["e1"],
        width=20, layer=MT2,
        cross_section=gf.cross_section.metal1,
        separation=30,
        steps=mySteps
    )
    for i in routes:
        c.add(i.references)
        
        
    mySteps = [{'dy': -130}, {'dx': -520}, {'dy': -50}, {'dx': -400}, {'dy': -690}, {'dx': 220}, {'dy': -40},
               {'dx': 100}, {'dy': -40}, {'dx': 100}, {'dy': -40}, {'dx': 100}, {'dy': -150}] if psL == 500 \
        else [{'dy': -130}, {'dx': -1220}, {'dy': -840}, {'dx': 350}, {'dy': -40}, {'dx': 220}, {'dy': -170}] if psL == 1500 \
        else [{'dy': -130}, {'dx': -680}, {'dy': -50}, {'dx': -240}, {'dy': -50}, {'dx': -psL/4}, {'dy': -640}, {'dx': psL/4}, {'dy': -40}, {'dx': 120}, {'dy': -40},
        {'dx': 150}, {'dy': -40}, {'dx': 200}, {'dy': -40}, {'dx': 100}, {'dy': -150}]
    routes = gf.routing.get_bundle_from_steps(
        DCpad.ports["e2"], TPM_act.ports["e2"],
        width=20, layer=MT2,
        cross_section=gf.cross_section.metal1,
        separation=30,
        steps=mySteps
    )
    for i in routes:
        c.add(i.references)

    # optical
    routes, lengths = [], []
    ports_list.append(mmi_i.ports["o1"])
    ports_list.append(mmi_i.ports["o2"])
    ports_list.append(mmi_o.ports["o1"])
    ports_list.append(mmi_o.ports["o2"])
    ports_list[0].name="o1_in"
    ports_list[1].name = "o2_in"
    ports_list[2].name = "o3_out"
    ports_list[3].name = "o4_out"

    # input GC to MMI
    if params["wsclDOEcell_placeGC"]:
        routes.append(gf.routing.get_bundle_from_steps(
            GC.ports["o3"], mmi_i.ports["o1"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=10,
            steps=[{'dy': 15}, {'dx': 2}])[0])
        routes.append(gf.routing.get_bundle_from_steps(
            GC.ports["o4"], mmi_i.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=10,
            steps=[{'dy': 25}, {'dx': -2}])[0])
    # MMI to TPM (collect lengths)
    routes.append(gf.routing.get_bundle_from_steps(
        mmi_i.ports["o3"], TPM_act.ports["o1"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dy': 15}, {'dx': -20}])[0])
    lengths.append(routes[-1].length)
    routes.append(gf.routing.get_bundle_from_steps(
        mmi_i.ports["o4"], TPM_idl.ports["o1"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dy': 15}])[0])
    lengths.append(routes[-1].length)
    # TPM to PS (collect lengths)
    routes.append(gf.routing.get_bundle_from_steps( #upper TPM
        TPM_idl.ports["o2"], device.ports["o2"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dx': 85 - psL / 2}])[0])
    lengths.append(routes[-1].length)
    routes.append(gf.routing.get_bundle_from_steps( #lower TPM
        TPM_act.ports["o2"], device.ports["o1"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dx': 72 - psL / 2}])[0])
    lengths.append(routes[-1].length)
    # PS to trombone (collect lengths)
    routes.append(gf.routing.get_bundle_from_steps(
        device.ports["o4"], trs.ports["o2"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dx': 20}])[0])
    lengths.append(routes[-1].length)
    routes.append(gf.routing.get_bundle_from_steps(
        device.ports["o3"], trs.ports["o4"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dx': 30}])[0])
    lengths.append(routes[-1].length)
    # trombone to MMI (collect lengths)
    routes.append(gf.routing.get_bundle_from_steps(
        trs.ports["o1"], mmi_o.ports["o3"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dy': -330}, {'dx': -50}])[0])
    lengths.append(routes[-1].length)
    routes.append(gf.routing.get_bundle_from_steps(
        trs.ports["o3"], mmi_o.ports["o4"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=10,
        steps=[{'dy': -340}, {'dx': -30}])[0])
    lengths.append(routes[-1].length)

    if params["wsclDOEcell_placeGC"]:
        # MMI to output GCs
        routes.append(gf.routing.get_bundle_from_steps(
            mmi_o.ports["o1"], GC.ports["o1"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=10,
            steps=[{'dy': -30}])[0])
        routes.append(gf.routing.get_bundle_from_steps(
            mmi_o.ports["o2"], GC.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=10,
            steps=[{'dy': -30}])[0])

    for i in routes:
        c.add(i.references)

    # check imbalance
    #print(lengths)
    myDeltaL = lengths[0] + lengths[2] + lengths[4] + lengths[6] - (lengths[1] + lengths[3] + lengths[5] + lengths[7])
    if myDeltaL + trombL != tgtImbalance:
        print("Warning: wscl1 MZI imbalance %f different from target %f" % (myDeltaL + trombL, tgtImbalance))

    ### termination of floated electrical ports of TO phase shifters
    term = c << gf.components.pad(size=(20, 3), layer=MT2, port_inclusion=0, port_orientation=90)
    term.connect("e2", getattr(TPM_idl, f"ports")["e1"])
    term = c << gf.components.pad(size=(20, 3), layer=MT2, port_inclusion=0, port_orientation=90)
    term.connect("e2", getattr(TPM_idl, f"ports")["e2"])

    # # LC dam
    # c.add_polygon([(-psL / 2 + 20, 0), (psL / 2 - 20, 0), (psL / 2 - 20, 50), (-psL / 2 + 20, 50)], layer=OXOP if damLayer == 'OXOP' else DTR).move((1170, 350))
    # c.add_polygon([(-psL / 2 + 20, 0), (psL / 2 - 20, 0), (psL / 2 - 20, 50), (-psL / 2 + 20, 50)], layer=OXOP if damLayer == 'OXOP' else DTR).move((1170, -400))

    # show die
    # c.show()"""
    '''
    d = merge_clad(c, 0)
    d.name = "wsclDOEcell"
    d.add_ports(ports_list)
    
    return d


def wsclDOEarray(default_params, doe_CSV, n_cols:int, pitch_x:float, name="array", rotation=0, GC_array_movex=6500, GC_array_movey=-1700):
    '''

    :param default_params:
    :param doe_CSV:
    :param n_cols:
    :param pitch_x:
    :param name:
    :param rotation:
    :param GC_array_movex:
    :param GC_array_movey:
    :return:

    example:
    array = wsclDOEarray(combined_params, None, n_cols = 3, pitch_x=2200, name="wsclDOEarray", rotation=0, GC_array_movex=3500, GC_array_movey=-3700)
    array.name = "wsclDOEarray"
    array.show()
    '''
    c = gf.Component()
    array, array_refs = PEO_place_array(wsclDOEcell, default_params, doe_CSV, n_rows=1, n_cols=n_cols, pitch_y=1000, pitch_x=pitch_x)
    c << array
    GC = c << GC_array_v3 (n_cols*4, 127)
    GC.rotate(90+rotation).mirror((1, 0)).move((GC_array_movex, GC_array_movey))

    GC_key_list = list(GC.ports.keys())
    GC_key_list.reverse()

    #merge ports from each wscl doe cell all into one component

    GC_key_list_subset = GC_key_list[1:5] #device 1
    GC_dict_subset = {k: GC[k] for k in GC_key_list_subset}
    routes = gf.routing.get_bundle_from_steps(
        array_refs[0].ports, GC_dict_subset,
        cross_section=rib_Oband,
        steps=[{'dy': -240}])
    for route in routes:
        c.add(route.references)

    GC_key_list_subset = GC_key_list[5:9] #device 2
    GC_dict_subset = {k: GC[k] for k in GC_key_list_subset}
    routes = gf.routing.get_bundle_from_steps(
        array_refs[1].ports, GC_dict_subset,
        cross_section=rib_Oband,
        steps=[{'dy': -240}])
    for route in routes:
        c.add(route.references)

    GC_key_list_subset = GC_key_list[9:13] #device 3
    GC_dict_subset = {k: GC[k] for k in GC_key_list_subset}
    routes = gf.routing.get_bundle_from_steps(
        array_refs[2].ports, GC_dict_subset,
        cross_section=rib_Oband,
        steps=[{'dy': -580}])
    for route in routes:
        c.add(route.references)

    routes = gf.routing.get_bundle_from_steps(
        GC.ports["o1"], GC.ports["o"+str(n_cols*4 + 2)],
        cross_section=rib_Oband,
        steps=[{'dy': 50},
               {'dx': -50},
               {'dy': -200},
               {'dx': -1600+127/2},
               {'dy': 200},
               {'dx': 50}])
    for route in routes:
        c.add(route.references)

    return c

def wsclDOEarray_v2(default_params, doe_CSV, n_cols:int, pitch_x:float, name="array", rotation=0, GC_array_movex=6500, GC_array_movey=-1700): #autoroute
    c = gf.Component()
    array, array_refs = PEO_place_array(wsclDOEcell, default_params, doe_CSV, n_rows=1, n_cols=n_cols, pitch_y=1000, pitch_x=pitch_x)
    c << array
    GC = c << GC_array_v3 (n_cols*4, 127)
    GC.rotate(90+rotation).mirror((1, 0)).move((GC_array_movex, GC_array_movey))

    GC_key_list = list(GC.ports.keys())
    GC_key_list.reverse()

    #merge ports from each wscl doe cell all into one component
    b = gf.Component("dummy_port_component")
    for i in range(0, n_cols):
        p = array_refs[i].ports["o1_in"]
        p.name= p.name + "_" + str(i)
        b.add_port(p)
        p = array_refs[i].ports["o2_in"]
        p.name= p.name + "_" + str(i)
        b.add_port(p)
        p = array_refs[i].ports["o3_out"]
        p.name= p.name + "_" + str(i)
        b.add_port(p)
        p = array_refs[i].ports["o4_out"]
        p.name= p.name + "_" + str(i)
        b.add_port(p)
    b.pprint_ports()
    dummy_port_ref = c << b

    GC_key_list_subset = GC_key_list[1:n_cols*4+1] #gcs for wscl in/out
    GC_dict_subset = {k: GC[k] for k in GC_key_list_subset}
    routes = gf.routing.get_bundle_from_steps(
        dummy_port_ref.ports, GC_dict_subset,
        cross_section=rib_Oband,
        steps=[{'dy': -240}])

    for route in routes:
        c.add(route.references)

    return c

'''
********************************************************* TEST STRUCTURES ****************************************************************
'''
@gf.cell
def ocdrWG(params:dict, slotW, wgOff, wgOff1, wgWin, wgWfin, wgtL, slW, oxW, oxOff, slWgL, dop1, dop2, dopOff, gapIND,
             xSlab, ribShape, ribPitch, ribDist, rep):
    #set parameters to internal names - prefer to call directly in future iterations
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]

    s2s_len_in = params["s2s_len_in"]
    s2s_width_in = params["s2s_width_in"]
    s2s_len_MMI = params["s2s_len_MMI"]
    s2s_width_MMI = params["s2s_width_MMI"]
    s2s_len_taper = params["s2s_len_taper"]
    s2s_gap_MMI_extra_slab = params["s2s_gap_MMI_extra_slab"]

    #LC_bound_width = params["LC_bound_width"]

    @gf.cell
    def taper(slotW, wgOff, wgOff1, wgWin, wgWfin, wgtL, xSlab, slW, oxW, oxOff, slWgL, dop1, dop2, dopOff):
        # generate taper region
        tap = gf.Component()
        # draw RIB
        myPol = [(0, 0), (0, 0.25), (wgOff, 0.25), (wgOff, wgWin), (wgOff + wgOff1, wgWin),
                 (wgOff + wgOff1 + wgtL, wgWfin), (wgOff + wgOff1 + wgtL, slotW / 2),
                 (wgOff + wgOff1, slotW / 2), (wgOff + wgOff1, 0)]
        myS = tap.add_polygon(myPol, layer=RIB)
        myS = tap.add_polygon(myPol, layer=RIB)
        myS.mirror((1, 0))

        # draw slab
        myPol = [(wgOff + wgOff1, wgWin - 0.3), (wgOff + wgOff1, wgWin), (wgOff + wgOff1 + wgtL, slW),
                 (wgOff + wgOff1 + wgtL, wgWfin - 0.09)]
        myS = tap.add_polygon(myPol, layer=SLAB)
        myS = tap.add_polygon(myPol, layer=SLAB)
        myS.mirror((1, 0))

        # draw oxOpen
        myPol = [(oxOff, 0), (oxOff, oxW / 2), (wgOff + wgOff1 + wgtL, oxW / 2), (wgOff + wgOff1 + wgtL, 0)]
        myS = tap.add_polygon(myPol, layer=OXOP)
        myS = tap.add_polygon(myPol, layer=OXOP)
        myS.mirror((1, 0))

        # draw dopings
        myPol = [(dopOff, 0), (dopOff, 4), (wgOff + wgOff1 + wgtL, 4), (wgOff + wgOff1 + wgtL, 0)]
        myS = tap.add_polygon(myPol, layer=NIM if dop1 == 'n' else PIM)
        myS = tap.add_polygon(myPol, layer=NIM if dop2 == 'n' else PIM)
        myS.mirror((1, 0))

        # draw extra slab if required
        if xSlab == 'y':
            OXSLAB_min_w = 0.4
            extra_slab_doping_w = 2
            extra_slab_doping_buffer_overlay = 0.1
            offset_extra_slab_doping = wgWin + s2s_gap_MMI_extra_slab + extra_slab_doping_w / 2 - extra_slab_doping_buffer_overlay
            myPol = [(wgOff / 2, wgWin + s2s_gap_MMI_extra_slab + OXSLAB_min_w / 2),
                     (wgOff / 2, wgWin + s2s_gap_MMI_extra_slab + OXSLAB_min_w * 3 / 2),
                     (wgOff / 2 + wgOff1 + wgtL, wgWin + s2s_gap_MMI_extra_slab + OXSLAB_min_w * 3 / 2),
                     (wgOff / 2 + wgOff1 + wgtL, wgWin + s2s_gap_MMI_extra_slab + OXSLAB_min_w / 2)]
            myS = tap.add_polygon(myPol, layer=SLAB)
            myS = tap.add_polygon(myPol, layer=SLAB)
            myS.mirror((1, 0))
            myPol = [(wgOff / 2, offset_extra_slab_doping - extra_slab_doping_w / 2),
                     (wgOff / 2, offset_extra_slab_doping + extra_slab_doping_w / 2),
                     (wgOff / 2 + wgOff1 + wgtL, offset_extra_slab_doping + extra_slab_doping_w / 2),
                     (wgOff / 2 + wgOff1 + wgtL, offset_extra_slab_doping - extra_slab_doping_w / 2)]
            myS = tap.add_polygon(myPol, layer=NIM if dop1 == 'n' else PIM)
            myS = tap.add_polygon(myPol, layer=NIM if dop1 == 'n' else PIM)
            myS.mirror((1, 0))

        return tap

    @gf.cell
    def ribCell(ribShape, ribDist):
        ribCell = gf.Component()
        rp1 = ribCell.add_polygon(ribShape, layer=RIB)
        rp1.move((0, ribDist + wgWfin))
        rp2 = ribCell.add_polygon(ribShape, layer=RIB)
        # rp2.mirror((1,0)).move((0,-(ribDist+wgWfin)))
        rp2.move((0, -(ribDist + wgWfin + rp2.ysize)))

        return ribCell

    # @gf.cell
    # def wgCell(slotW, wgOff, wgOff1, wgWin, wgWfin, wgtL, xSlab, slW, oxW, oxOff, slWgL, dop1, dop2, dopOff, gapIND,
    #            ribShape, ribPitch, ribDist): #obsolete, AMF process, GLM style of placement
    #
    #     wgCell = gf.Component()
    #     # place two tapers
    #     myS = wgCell.add_ref(
    #         taper(slotW, wgOff, wgOff1, wgWin, wgWfin, wgtL, xSlab, slW, oxW, oxOff, slWgL, dop1, dop2, dopOff))
    #     myS = wgCell.add_ref(
    #         taper(slotW, wgOff, wgOff1, wgWin, wgWfin, wgtL, xSlab, slW, oxW, oxOff, slWgL, dop1, dop2, dopOff))
    #     myS.mirror((0, 1)).movex(slWgL + 2 * (wgOff + wgOff1 + wgtL))
    #
    #     # place slot waveguide
    #     w_slotWG = wgWfin - slotW / 2
    #     w_slot = slotW
    #
    #     offset_slotWG = (w_slotWG + w_slot) / 2
    #     s1 = gf.Section(width=w_slotWG, offset=offset_slotWG, layer=RIB, name="slotWG_1")
    #     s2 = gf.Section(width=w_slotWG, offset=-offset_slotWG, layer=RIB, name="slotWG_2")
    #
    #     offset_slab = (w_slot + w_slab) / 2 + w_slotWG
    #     s3 = gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=offset_slab, layer=SLAB, name="slab_1")
    #     s4 = gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=-offset_slab, layer=SLAB, name="slab_2")
    #
    #     offset_si_contact = w_slot / 2 + w_slotWG + gap_si_contact + w_si_contact / 2
    #     s5 = gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact_1")
    #     s6 = gf.Section(width=w_si_contact, offset=-offset_si_contact, layer=RIB, name="sl_contact_2")
    #
    #     offset_MT1 = oxW / 2 + w_MT1 / 2 + min_gap_OXOP_MT
    #     s7 = gf.Section(width=w_MT1, offset=offset_MT1, layer=MT1, name="MT1_1")
    #     s8 = gf.Section(width=w_MT1, offset=-offset_MT1, layer=MT1, name="MT1_2")
    #
    #     offset_MT2 = oxW / 2 + w_MT1 / 2 + min_gap_OXOP_MT
    #     s72 = gf.Section(width=w_MT1, offset=offset_MT2, layer=MT2, name="MT2_1")
    #     s82 = gf.Section(width=w_MT1, offset=-offset_MT2, layer=MT2, name="MT2_2")
    #
    #     offset_via_1 = oxW / 2 + min_gap_OXOP_MT + min_inc_via_1 + via_size / 2
    #     s9v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA1), spacing=via_size + gap_via_1,
    #                              padding=via_size, enclosure=min_inc_via_1, offset=offset_via_1)
    #     s10v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA1), spacing=via_size + gap_via_1,
    #                               padding=via_size, enclosure=min_inc_via_1, offset=-offset_via_1)
    #
    #     offset_via_2 = oxW / 2 + min_gap_OXOP_MT + min_inc_via_1 + via_size + min_exc_of_via_2 + via_size / 2
    #     s11v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA2), spacing=min_exc_of_via_2 + via_size,
    #                               padding=via_size, offset=offset_via_2)
    #     s12v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA2), spacing=min_exc_of_via_2 + via_size,
    #                               padding=via_size, offset=-offset_via_2)
    #
    #     offset_NCONT = w_slot / 2 + w_slotWG + gap_NCONT_WG + w_NCONT / 2
    #     s13 = gf.Section(width=w_NCONT, offset=offset_NCONT, layer=NCONT if dop2 == 'n' else PCONT, name="NCONT_1")
    #     s14 = gf.Section(width=w_NCONT, offset=-offset_NCONT, layer=NCONT if dop1 == 'n' else PCONT, name="NCONT_2")
    #
    #     offset_IND = w_slot / 2 + w_slotWG + gapIND + w_IND / 2
    #     s15 = gf.Section(width=w_IND, offset=offset_IND, layer=IND if dop2 == 'n' else IPD, name="IND_1")
    #     s16 = gf.Section(width=w_IND, offset=-offset_IND, layer=IND if dop1 == 'n' else IPD, name="IND_2")
    #
    #     offset_NIM = w_NIM / 4
    #     s17 = gf.Section(width=w_NIM / 2, offset=offset_NIM, layer=NIM if dop2 == 'n' else PIM, name="NIM_1")
    #     s18 = gf.Section(width=w_NIM / 2, offset=-offset_NIM, layer=NIM if dop1 == 'n' else PIM, name="NIM_2")
    #
    #     s151 = gf.Section(width=oxW, layer=OXOP, name="oxide_open")
    #
    #     # print(ribShape, ribPitch)
    #     if ribShape != []:
    #         s1r = ComponentAlongPath(component=ribCell(ribShape, ribDist), spacing=ribPitch)
    #         x1 = gf.CrossSection(
    #             sections=(s1, s2, s3, s4, s5, s6, s7, s8, s72, s82, s13, s14, s15, s16, s17, s18, s151),
    #             components_along_path=[s9v, s10v, s11v, s12v, s1r]
    #         )
    #     else:
    #         x1 = gf.CrossSection(
    #             sections=(s1, s2, s3, s4, s5, s6, s7, s8, s72, s82, s13, s14, s15, s16, s17, s18, s151),
    #             components_along_path=[s9v, s10v, s11v, s12v]
    #         )
    #
    #     p1 = gf.path.straight(length=slWgL)
    #     p2 = gf.path.extrude(p1, x1)
    #     myS = wgCell.add_ref(p2)
    #     myS.movex(wgOff + wgOff1 + wgtL)
    #     # add ports
    #     wgCell.add_port(name='o1', center=(0, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")
    #     wgCell.add_port(name='o2', center=(wgCell.xsize, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")
    #     return wgCell
    #
    # c = gf.Component()
    #
    # # add cascade
    # a = wgCell(slotW, wgOff, wgOff1, wgWin, wgWfin, wgtL, xSlab, slW, oxW, oxOff, slWgL, dop1, dop2, dopOff, gapIND,
    #            ribShape, ribPitch, ribDist)
    # # c.add_ref(a,columns=rep, spacing=(a.xsize,0))
    # mySeg = []
    # for i in range(rep):
    #     mySeg.append(c.add_ref(a))
    #     myPos = (a.xsize + 1.0) * i + round(((0.5 if i > 0 else 0.5) - 0.5) * 1.0, 3)
    #     #myPos = (a.xsize + 1.0) * i + round(((random.random() if i > 0 else 0.5) - 0.5) * 1.0, 3)
    #     mySeg[-1].move((myPos, 0))
    # # connect
    # for i in range(1, rep):
    #     route = gf.routing.get_bundle_from_steps(
    #         mySeg[i - 1].ports["o2"], mySeg[i].ports["o1"],
    #         width=0.41, cross_section=rib_Oband)[0]
    #     c.add(route.references)
    #
    # # if rep>1 add metal connections
    # if rep > 1:
    #     offset_MT2 = oxW / 2 + min_gap_OXOP_MT
    #     c.add_polygon(
    #         [(0, offset_MT2), (c.xsize, offset_MT2), (c.xsize, offset_MT2 + w_MT1), (0, offset_MT2 + w_MT1)],
    #         layer=MT1)
    #     c.add_polygon([(0, -offset_MT2), (c.xsize, -offset_MT2), (c.xsize, -(offset_MT2 + w_MT1)),
    #                    (0, -(offset_MT2 + w_MT1))], layer=MT1)
    #
    # # add ports
    # c.add_port(name="o1", center=(0, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")
    # c.add_port(name="o2", center=(myPos + a.xsize, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")
    # c.add_port(name="e1", center=(wgOff + wgOff1 + wgtL, 12), width=0.5, orientation=0, layer=MT1,
    #            port_type="electrical")
    # c.add_port(name="e2", center=(wgOff + wgOff1 + wgtL, -12), width=0.5, orientation=0, layer=MT1,
    #            port_type="electrical")
    # c.add_port(name="e3", center=(myPos + a.xsize, 12), width=0.5, orientation=0, layer=MT1, port_type="electrical")
    # c.add_port(name="e4", center=(myPos + a.xsize, -12), width=0.5, orientation=0, layer=MT1,
    #            port_type="electrical")
    #
    #
    # return c

@gf.cell
def OCDR_DOEcell_from_csv_params(combined_params: dict, ocdrDOE: str, wg_L = 6): #requires DOE_pars.csv in same folder, change?
    '''
    returns 10 rows of ocdrWG structure with edge couplers and DC pads

    example:
    combined_params = {**DOE6_params, **electrode_DOE6_params, **differential_electrode_params, **balun_sipho_params}
    DOEocdr1=c<<OCDR_DOEcell_from_csv_params(combined_params, "DOE6_CSV_Params/OCDR_DOE1_pars.csv")
    :param combined_params:
    :param ocdrDOE:
    :param wg_L:
    :return:
    '''
    DC_pad_size_x = combined_params["DC_pad_size_x"]
    DC_pad_size_y = combined_params["DC_pad_size_y"]
    # THROW ERROR IF CORRECT PARAMS NOT FOUND
    #fid = open(ocdrDOE + '_log.txt', 'a')

    c = gf.Component(ocdrDOE)

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place IOs
    EC_start_position = 0
    EC_length = 370
    EC_N = 5
    EC_1 = c << EC_array(EC_N, EC_pitch)
    EC_1.rotate(270).move((EC_length + 100, EC_start_position))
    EC_2 = c << EC_array(EC_N, EC_pitch)
    EC_2.rotate(270).move((EC_length + 100, EC_start_position + 127 * 6))

    # place pads
    DC_pad_start_position_y = -445
    DC = []
    DC.append(c << DC_pads(DC_pad_size_x, DC_pad_size_y))
    DC[0].rotate(180).move((700, DC_pad_start_position_y))

    # place waveguides
    # first cell is just a regular waveguide with OX opens
    # place oxide opens
    # then open DOE parameters and generate waveguides
    pitchY = 50
    doe = pd.read_csv(ocdrDOE)
    sw, device, routes = [], [], []
    for i, j in doe.iterrows():
        # skip cells
        if i < 10:
            device.append(c << PS_connected_from_params(combined_params))

                          # ocdrWG(params = combined_params, slotW=doe['slotW(um)'][i], wgOff=doe['wgOff(um)'][i], wgOff1=doe['wgOff1(um)'][i],
                          #             wgWin=doe['wgWin(um)'][i], wgWfin=doe['wgWfin(um)'][i], wgtL=doe['wgtL(um)'][i],
                          #             xSlab='n',
                          #             slW=doe['slW(um)'][i], oxW=doe['oxW(um)'][i], oxOff=doe['oxOff(um)'][i],
                          #             slWgL=doe['slWgL(um)'][i],
                          #             dop1=doe['dop1'][i], dop2=doe['dop2'][i], dopOff=doe['dopOff(um)'][i],
                          #             gapIND=doe['gapIND(um)'][i],
                          #             ribShape=list(eval(doe['ribShape'][i])), ribPitch=doe['ribPitch(um)'][i],
                          #             ribDist=doe['ribDist(um)'][i], rep=1))
            device[i].move((1000, (i * pitchY if i < 5 else (i + 1) * pitchY) - pitchY * 3))

            #add WG extension
            my_Center = device[i].ports["o2"].center
            my_Center[0] = my_Center[0] + wg_L
            _ = c.add_port(name="o3_"+str(i), center=(my_Center),width=0.41, orientation=180, cross_section=rib_Oband, port_type="optical")
            # print(_.center)
            # print(device[i].ports["o2"].center)
            routes.append(gf.routing.get_bundle(
                _,device[i].ports["o2"],
                width=0.41, cross_section=rib_Oband,
                separation=5, radius=15)[0])
            #device[i].move((1000, (i * 50 if i < 5 else (i + 1) * 50) - 50 * 4))
    for i in routes:
        c.add(i.references)

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    # optical
    routes = []
    for i in range(5):
        routes.append(gf.routing.get_bundle_from_steps(
            EC_1.ports["o%d" % (5 - i)], device[i].ports["o1"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dx': 130 + (5 - i) * 10}, {'dy': -10}])[0])
        routes.append(gf.routing.get_bundle_from_steps(
            EC_2.ports["o%d" % (5 - i)], device[i + 5].ports["o1"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dx': 130 + i * 10}, {'dy': -10}])[0])
    for i in routes:
        c.add(i.references)

    # electrical
    routes = []
    for i in range(5):
        routes.append(gf.routing.get_bundle_from_steps(
            DC[0].ports["e1"], device[i].ports["e1"],
            cross_section=gf.cross_section.metal1,
            width=8, layer=MT1)[0])
        routes.append(gf.routing.get_bundle_from_steps(
            DC[0].ports["e1"], device[i + 5].ports["e1"],
            cross_section=gf.cross_section.metal1,
            width=8, layer=MT1)[0])
        routes.append(gf.routing.get_bundle_from_steps(
            DC[0].ports["e2"], device[i].ports["e2"],
            cross_section=gf.cross_section.metal1,
            width=8, layer=MT2)[0])
        routes.append(gf.routing.get_bundle_from_steps(
            DC[0].ports["e2"], device[i + 5].ports["e2"],
            cross_section=gf.cross_section.metal1,
            # steps=[{'dy':120},{'dx':400}] if i==4 else [],
            width=8, layer=MT2)[0])

    for i in routes:
        c.add(i.references)

    # add vias
    c.add_polygon([(700 - 1.5, DC_pad_start_position_y - 66 + 140),
                 (700 + 3 - 1.5, DC_pad_start_position_y - 66 + 140),
                 (700 + 3 - 1.5, DC_pad_start_position_y + 3 - 66 + 140),
                 (700 - 1.5, DC_pad_start_position_y + 3 - 66 + 140)], layer=VIA2)
    c.add_polygon([(700 - 1.5, DC_pad_start_position_y - 5 - 66 + 140),
                   (700 + 3 - 1.5, DC_pad_start_position_y - 5 - 66 + 140),
                   (700 + 3 - 1.5, DC_pad_start_position_y + 3 - 5 - 66 + 140),
                   (700 - 1.5, DC_pad_start_position_y + 3 - 5 - 66 + 140)], layer=VIA2)
    c.add_polygon([(700 - 1.5, DC_pad_start_position_y - 10 - 66 + 140),
                   (700 + 3 - 1.5, DC_pad_start_position_y - 10 - 66 + 140),
                   (700 + 3 - 1.5, DC_pad_start_position_y + 3 - 10 - 66 + 140),
                   (700 - 1.5, DC_pad_start_position_y + 3 - 10 - 66 + 140)], layer=VIA2)
    c.add_polygon(
        [(700 - 10, DC_pad_start_position_y - 58 - 22 + 140), (700 + 10, DC_pad_start_position_y - 58 - 22 + 140),
         (700 + 10, DC_pad_start_position_y + 20 - 58 - 22 + 140),
         (700 - 10, DC_pad_start_position_y + 20 - 58 - 22 + 140)], layer=MT2)

    # c.add_polygon([(700-1.5,DC_pad_start_position_y+10), (700+3-1.5,DC_pad_start_position_y+10), (700+3-1.5,DC_pad_start_position_y+3+10), (700-1.5,DC_pad_start_position_y+3+10)], layer=VIA2)

    # show die
    # c.show()
    # fid.close()
    return c

@gf.cell
def TC_DOE_new(combined_params:dict, TCdoe: str, row: int = 1):
    '''create repeating taper structure for testing purposes

    example:

    c = gf.Component("TC_DOE_w_electrical")

    combined_params = {**balun_electrode_params, **balun_sipho_params}
    TC_params = combined_params
    TC_params["electrical"] = True # can be True or False
    TC_params["MT1_from_PS"] = False #each PS has M1 contacts
    TC_params["min_gap_OXOP_MT"] = -2

    tc=c<<TC_DOE_new(TC_params,'DOE6_CSV_Params/TC_DOE_pars.csv',3)

    c.show()
    '''

    @gf.cell
    def s2s_cascade_column(params: dict, rep):
        c = gf.Component()

        # add cascade
        cascade_PS_length = 0
        if combined_params["electrical"]:
            cascade_PS_length = 30
            ps_params = {**combined_params, "PS_length": cascade_PS_length }
            a = PS_connected_from_params(ps_params)
        else:
            a = PS_connected_s2s_only(params, Oband_variant=True)

        mySeg = []
        for i in range(rep):
            mySeg.append(c.add_ref(a))
            # myPos = (a.xsize + 1.0) * i + round(((0.5 if i > 0 else 0.5) - 0.5) * 1.0, 3)
            myPos = (a.xsize + 1.0) * i + round(((random.random() if i > 0 else 0.5) - 0.5) * 1.0, 3)
            mySeg[-1].move((myPos, 0))

        # connect
        for i in range(1, rep):
            route = gf.routing.get_bundle_from_steps(
                mySeg[i - 1].ports["o1"], mySeg[i].ports["o2"],
                width=params["s2s_O_w_in"], cross_section=rib_Oband)[0]
            c.add(route.references)

        # if rep>1 add metal connections
        if params["electrical"] and rep > 1:
            offset_MT1 = params["w_OXOP"] / 2 + params["min_gap_OXOP_MT"] + params["w_MT1"] / 2
            w_MT1 = params["w_MT1"]
            c.add_polygon([(0, offset_MT1),
                           (c.xsize- params["s2s_O_len_taper"], offset_MT1),
                           (c.xsize-params["s2s_O_len_taper"], offset_MT1 + w_MT1),
                           (0, offset_MT1 + w_MT1)],
                layer=MT1)
            c.add_polygon([(0, -offset_MT1),
                           (c.xsize-28.38, -offset_MT1),
                           (c.xsize-28.38, -(offset_MT1 + w_MT1)),
                           (0, -(offset_MT1 + w_MT1))],
                          layer=MT1)

        c.add_polygon([(c.xmin, c.ymin),
                       (c.xmax, c.ymin),
                       (c.xmax, c.ymax),
                       (c.xmin, c.ymax)],
                      layer=NOP)

        c.add_polygon([(c.xmin, c.ymin),
                       (c.xmax, c.ymin),
                       (c.xmax, c.ymax),
                       (c.xmin, c.ymax)],
                      layer=RIB_ETCH)

        # add ports
        c.add_port(name="o1", center=(0, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")
        c.add_port(name="o2", center=(myPos + a.xsize - cascade_PS_length -2.1, 0), width=params["s2s_O_w_in"], orientation=0, layer=RIB, port_type="optical")

        if params["electrical"]:
            c.add_port(name="e1", center=(15, offset_MT1), width=0.5, orientation=0, layer=MT1,
                       port_type="electrical")
            c.add_port(name="e2", center=(15, -2 - offset_MT1), width=0.5, orientation=0, layer=MT1,
                       port_type="electrical")
            c.add_port(name="e3", center=(myPos + a.xsize -params["s2s_O_len_taper"], 2.5 + offset_MT1), width=2*w_MT1, orientation=0, layer=MT1, port_type="electrical")
            c.add_port(name="e4", center=(myPos + a.xsize -params["s2s_O_len_taper"], -2.5 - offset_MT1), width=2*w_MT1, orientation=0, layer=MT1,
                       port_type="electrical")

        return c

    DC_pad_size_x = combined_params["DC_pad_size_x"]
    DC_pad_size_y = combined_params["DC_pad_size_y"]
    min_gap_OXOP_MT = combined_params["min_gap_OXOP_MT"]
    w_MT1 = combined_params["w_MT1"]
    # fid=open(TCdoe+'_log.txt','a')
    c = gf.Component()

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place IOs
    EC_start_position = 0
    EC_length = 370
    EC_N = 2
    EC_1 = c << EC_array(EC_N, EC_pitch)
    EC_1.rotate(270).move((EC_length + 100, EC_start_position))

    # place pads
    if combined_params["electrical"]:
        DC_pad_start_position_y = -600
        DC = []
        DC.append(c << DC_pads(DC_pad_size_x, DC_pad_size_y))
        DC[0].rotate(180).move((1400, DC_pad_start_position_y))

    # place cascade waveguides. TC DOE has a single entry
    doe = pd.read_csv(TCdoe)
    sw, device = [], []
    i = row - 1
    for column_name in doe.columns: #overwrite existing params with csv params
        combined_params = {**combined_params, column_name: doe[column_name][i]}

    a = s2s_cascade_column(params=combined_params, rep=int(doe['rep'][i]))
    device.append(c.add_ref(a))
    device.append(c.add_ref(a))
    device[0].rotate(90).move((1450, 100 if doe['rep'][i] > 6 else 300))
    device[1].rotate(90).move(# (doe['w_OXOP(um)'][i] + min_gap_OXOP_MT * 2 + w_MT1 * 2, 0))
     #(1450 + doe['w_OXOP(um)'][i] + min_gap_OXOP_MT * 2 + w_MT1 * 2, 100 if doe['rep'][i] > 6 else 300))
    (1450 + doe['w_OXOP(um)'][i] +min_gap_OXOP_MT + 3.5* w_MT1, 100 if doe['rep'][i] > 6 else 300))

    # routing
    # optical
    routes = []
    for i in range(1):
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["o1"], EC_1.ports["o1"],
            width=0.5, cross_section = rib,
            separation=5, radius=15,
            # steps=[{'dy':20},{'dx':-10}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["o2"], device[1].ports["o2"],
            width=combined_params["s2s_O_w_in"], cross_section = rib,
            separation=5, radius=15,
            steps=[{'dy': 20}, {'dx': -20}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            device[1].ports["o1"], EC_1.ports["o2"],
            width=0.5, cross_section = rib,
            separation=5, radius=15,
            # steps=[{'dy':20},{'dx':-20},{'dy':-20}]
        )[0])

    for i in routes:
        c.add(i.references)


    #electrical
    if combined_params["electrical"]:
        routes = []
        for i in range(1):
            routes.append(gf.routing.get_bundle_from_steps(
                device[0].ports["e1"], DC[0].ports["e2"],
                cross_section=gf.cross_section.metal1,
                width=8, layer=MT2,
                steps=[{'dy': -20}, {'dx': -50}, {'dy': -200}, {'dx': -600}, {'dy': -300}]
            )[0])
            routes.append(gf.routing.get_bundle_from_steps(
                device[0].ports["e2"], DC[0].ports["e1"],
                cross_section=gf.cross_section.metal1,
                width=8, layer=MT2,
                steps=[{'dy': -60}, {'dx': -50}, {'dy': -200}, {'dx': -500}, {'dy': -200}]
            )[0])
            routes.append(gf.routing.get_bundle_from_steps(
                device[0].ports["e3"], device[1].ports["e4"],
                cross_section=gf.cross_section.metal1,
                width=8, layer=MT1,
                steps=[{'dy': 30}, {'dx': 10}]
            )[0])

        for i in routes:
            c.add(i.references)


    return c



@gf.cell
def TC_DOEcell_from_csv_params(combined_params:dict, TCdoe: str, row: int = 1):
    """old version"""
    DC_pad_size_x = combined_params["DC_pad_size_x"]
    DC_pad_size_y = combined_params["DC_pad_size_y"]
    min_gap_OXOP_MT = combined_params["min_gap_OXOP_MT"]
    w_MT1 = combined_params["w_MT1"]
    #THROW ERROR IF CORRECT PARAMS NOT FOUND
    # fid=open(TCdoe+'_log.txt','a')
    c = gf.Component()

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place IOs
    EC_start_position = 0
    EC_length = 370
    EC_N = 2
    EC_1 = c << EC_array(EC_N, EC_pitch)
    EC_1.rotate(270).move((EC_length + 100, EC_start_position))

    # place pads
    DC_pad_start_position_y = -600
    DC = []
    DC.append(c << DC_pads(DC_pad_size_x, DC_pad_size_y))
    DC[0].rotate(180).move((1400, DC_pad_start_position_y))

    # place cascade waveguides. TC DOE has a single entry
    doe = pd.read_csv(TCdoe)
    sw, device = [], []
    i = row - 1
    a = ocdrWG(params=combined_params, slotW=doe['slotW(um)'][i], wgOff=doe['wgOff(um)'][i], wgOff1=doe['wgOff1(um)'][i],
               wgWin=doe['wgWin(um)'][i], wgWfin=doe['wgWfin(um)'][i], wgtL=doe['wgtL(um)'][i], xSlab=doe['xSlab'][i],
               slW=doe['slW(um)'][i], oxW=doe['oxW(um)'][i], oxOff=doe['oxOff(um)'][i], slWgL=doe['slWgL(um)'][i],
               dop1=doe['dop1'][i], dop2=doe['dop2'][i], dopOff=doe['dopOff(um)'][i], gapIND=doe['gapIND(um)'][i],
               ribShape=list(eval(doe['ribShape'][i])), ribPitch=doe['ribPitch(um)'][i], ribDist=doe['ribDist(um)'][i],
               rep=int(doe['rep'][i]))
    device.append(c.add_ref(a))
    device.append(c.add_ref(a))
    device[0].rotate(90).move((1450, 100 if doe['rep'][i] > 6 else 300))
    device[1].rotate(90).move(
        (1450 + doe['oxW(um)'][i] + min_gap_OXOP_MT * 2 + w_MT1 * 2, 100 if doe['rep'][i] > 6 else 300))

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    # optical
    routes = []
    for i in range(1):
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["o1"], EC_1.ports["o1"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            # steps=[{'dy':20},{'dx':-10}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["o2"], device[1].ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dy': 20}, {'dx': -20}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            device[1].ports["o1"], EC_1.ports["o2"],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            # steps=[{'dy':20},{'dx':-20},{'dy':-20}]
        )[0])
    for i in routes:
        c.add(i.references)

    # electrical
    routes = []
    for i in range(1):
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["e1"], DC[0].ports["e2"],
            cross_section=gf.cross_section.metal1,
            width=8, layer=MT2,
            steps=[{'dy': -20}, {'dx': -50}, {'dy': -200}, {'dx': -600}, {'dy': -300}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["e2"], DC[0].ports["e1"],
            cross_section=gf.cross_section.metal1,
            width=8, layer=MT2,
            steps=[{'dy': -60}, {'dx': -50}, {'dy': -200}, {'dx': -500}, {'dy': -200}]
        )[0])
        routes.append(gf.routing.get_bundle_from_steps(
            device[0].ports["e3"], device[1].ports["e4"],
            cross_section=gf.cross_section.metal1,
            width=8, layer=MT1,
            steps=[{'dy': 30}, {'dx': 10}]
        )[0])

    for i in routes:
        c.add(i.references)

    # c.add_polygon([(700-1.5,DC_pad_start_position_y+10), (700+3-1.5,DC_pad_start_position_y+10), (700+3-1.5,DC_pad_start_position_y+3+10), (700-1.5,DC_pad_start_position_y+3+10)], layer=VIA2)

    # show die
    # c.show()
    # fid.close()
    return c


# this defines the GCDOEcell: parameters are read from file
# rev 1.0 - 30Dec2024
@gf.cell
def GCDOEcell(gcDOEfile, l):
    # fid=open('DOElog.txt','a')
    doe = pd.read_csv(gcDOEfile + '_pars.csv')

    c = gf.Component()
    a = gf.Section(width=1, layer=RIB)
    b = gf.Section(width=1, layer=RIB)
    xs = gf.CrossSection(sections=(a, b))

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    for i, j in doe.iterrows():
        myGC = gf.components.grating_coupler_elliptical(polarization='te',
            taper_length=j['taperLength'], taper_angle=j['taperAngle'], wavelength=j['wl'],
            fiber_angle=j['fiberAngle'], grating_line_width=j['width'], neff=j['neff'],
            nclad=j['nclad'], n_periods=j['periods'], big_last_tooth=j['blt'],
            layer_slab=GRAT if j['slab'] == 'GRAT' else SLAB, slab_xmin=j['slabStart'], slab_offset=j['slabOff'],
            spiked=j['spiked'], cross_section=xs)

        # add slab layer regardless
        myGCSLAB = gf.components.grating_coupler_elliptical(polarization='te',
        taper_length=j['taperLength'], taper_angle=j['taperAngle'], wavelength=j['wl'],
        fiber_angle=j['fiberAngle'], grating_line_width=j['width'], neff=j['neff'],
        nclad=j['nclad'], n_periods=j['periods'], big_last_tooth=j['blt'],
        layer_slab=SLAB, slab_xmin=j['slabStart'], slab_offset=j['slabOff'],
        spiked=j['spiked'], cross_section=xs)

        _ = c << myGCSLAB
        _.rotate(270).move((0 + i // l * 170, (l - i % l - 1) * 60))
        myGC1 = c.add_ref(myGC)
        myGC1.rotate(270).move((0 + i // l * 170, (l - i % l - 1) * 60))

        _ = c << myGCSLAB
        _.rotate(270).move((127 + i // l * 170, (l - i % l - 1) * 60))
        myGC2 = c.add_ref(myGC)
        myGC2.rotate(270).move((127 + i // l * 170, (l - i % l - 1) * 60))
        # place label
        myLbl = c << gf.components.text('G ' + gcDOEfile[-1] + ' _ ' + ' '.join(str(i)), layer=MT2, size=20)
        myLbl.move((0 + i // l * 170 + 2, (l - i % l - 1) * 60 - 10))

        myRoute = gf.routing.get_bundle_from_steps(
            myGC1.ports['o1'], myGC2.ports['o1'],
            width=0.41, cross_section=rib_Oband,
            separation=5, radius=15,
            steps=[{'dy': 11}]
        )
        for i in myRoute:
            c.add(i.references)

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************

    # show die
    # c.show()
    # fid.close()
    return c


@gf.cell
def MRM_SilTerra(params: dict) -> gf.Component:
    """
    new version so that parameters can easily be read from CSV and placed in batches
    :param params:
    :param trans_length:
    :param s2s_type: can be "FNG" (default), "oxide", "extra_slab"
    :param extra_slab:
    :param Oband_variant:

    The below are overrides and specifications for GSGSG_MT2
    :param electrode_params

    :param taper_type: s2s overrides, 1 or 2
    :param sig_trace: ps overrides, "narrow" "medium" "wide"
    :param ps_config: ps parameter overrides: "default"(default), "narrow_custom" "medium_custom" "wide_custom":
    :param termination:
    if termination = 0, the terminating electrode pads will match the default pads for a symmetric electrode structure
            if termination = 35, 50, 65, output pad geometry will be changed to the specified resistance. Also, heaters between the termination pads will be added.
                also, these additional parameters must be specified in parameter dictionary: "htr_width_x", "htr_length", "htr_connect_length",
                                                                                            "sc_length", "pad_t_length",  "trans_length_to_term"
    :param gnds_shorted: bool, if True ground shorting structure will be added
    :param gsgsg_variant: almost obsolete, only used to specify SGS_MT2_DC in DOE8
    :param config: standard (default), batch, compact - used only in SGS_MT2_DC
    :return:

    example:
    c = gf.Component("Mirach_diff_MZM_GSGSG_taper_sample_2025_12_05")
    combined_params = {**differential_electrode_params, **balun_sipho_params,  "PS_trans_length": 250}
    combined_params["PS_length"] = 1000
    _ = c << MZM_SilTerra(combined_params)
    #_.rotate(-90)
    c.show()

    c = gf.Component("Mirach_diff_MZM_GSGSG_s2s_adiabatic_sample_2025_12_05")
combined_params = {**differential_electrode_params, **balun_sipho_params,  "MT1_from_PS": False, "PS_trans_length": 250, "PS_taper": True, "DC_MT1": True, "s2s_type": "adiabatic",
                   "W": 0.4,
                   "W0": 0.13,
                   "R": 0.18,
                   "S": 0.16,
                   "G": 0.16,
                   "L1": 4,
                   "L2": 4,
                   "L3": 8,
                   "B1": 2.5,
                   "B2": 0.9,
                   "B3": 2,
                   "C1": 0.5,
                   "C2": 0.13,
            }
combined_params["PS_length"] = 600
_ = c << MZM_SilTerra(combined_params)
#_.rotate(-90)
c.show()
    """
    #imports for legacy sub-functions
    w_routing = params["w_routing"]
    w_FETCH_CLD = params["w_FETCH_CLD"]
    
    trans_length = params["trans_length"]
    taper_type = params["taper_type"]
    sig_trace = params["sig_trace"]
    termination = params["termination"]
    extra_slab = params["extra_slab"]
    config = params["config"]
    gsgsg_variant = params["gsgsg_variant"]
    Oband_variant = params["Oband_variant"]
    s2s_type  = params["s2s_type"]
    ps_config = params["ps_config"]
    gnds_shorted = params["gnds_shorted"]
    
    coupling_length = 48
    coupling_gap = 0.25
    bus_length = coupling_length
    bus_extend = 50

    bend_r = 15
    
    c = gf.Component()

    ref_bus = c << gf.components.straight(length=bus_length, cross_section = rib_Oband)
    ref_coupling = c << gf.components.straight(length=coupling_length, cross_section = rib_Oband)
    ref_coupling.movey(coupling_gap+w_routing)
 
    ref_bend_1 = c << gf.components.bend_euler(radius=bend_r, angle=90, npoints=40, cross_section = rib_Oband)
    ref_bend_1.connect('o1', ref_coupling.ports["o2"])
    ref_bend_4 = c << gf.components.bend_euler(radius=bend_r, angle=90, npoints=40, cross_section = rib_Oband)
    ref_bend_4.connect('o2', ref_coupling.ports["o1"])

    ref_bus_ext1 = c << gf.components.straight(length=bus_length, cross_section = rib_Oband)
    ref_bus_ext1.connect('o1', ref_bus.ports["o2"])
    ref_bus_ext2 = c << gf.components.straight(length=bus_length, cross_section = rib_Oband)
    ref_bus_ext2.connect('o2', ref_bus.ports["o1"])   

    
    # Create two PS_connected
    PS1 = PS_connected_from_params(params)
    ref_PS_1 = c << PS1
    ref_PS_1.connect("o1", ref_bend_1["o2"])

    PS2 = PS_connected_from_params(params)
    ref_PS_2 = c << PS2
    ref_PS_2.connect("o2", ref_bend_4["o1"])

    ref_bend_2 = c << gf.components.bend_euler(radius=bend_r, angle=90, npoints=40, cross_section = rib_Oband)
    ref_bend_2.connect('o1', ref_PS_1.ports["o2"])
 
    ref_bend_3 = c << gf.components.bend_euler(radius=bend_r, angle=90, npoints=40, cross_section = rib_Oband)
    ref_bend_3.connect('o2', ref_PS_2.ports["o1"])

    routes = gf.routing.get_bundle(
        ref_bend_2.ports["o2"],
        ref_bend_3.ports["o1"],
        cross_section=rib_Oband
    )
    for route in routes:
        c.add(route.references)

    c = merge_clad(c, 0)

    # # Choose GSGSG implementation
    # PS_length = params["PS_length"]
    # if gsgsg_variant == "DC":
    #     # DOE8 specific DC variant
    #     GSGSG = SGS_MT2_DC(PS_length, trans_length, taper_type, sig_trace, config=config, params=params)
    # # elif gsgsg_variant == "bond":
    # #     GSGSG = GSGSG_MT2_symmetric(PS_length, trans_length, taper_type, sig_trace, electrode_params, termination=termination, ps_config=ps_config, gnds_shorted=True)
    # else:
    #     # Standard GSGSG variant
    #     GSGSG = GSGSG_MT2(PS_length, trans_length, taper_type, sig_trace, params=params, termination=termination, gnds_shorted = gnds_shorted, ps_config = ps_config)

    # ref_SGS = c << GSGSG

    # # Connect electrical ports
    # ref_PS_1.connect("e_MT2", ref_SGS.ports["e_up"])
    # ref_PS_2.connect("e_MT2", ref_SGS.ports["e_low"])

    # Add optical ports
    c.add_port("o1", port=ref_bus_ext2["o1"])
    c.add_port("o2", port=ref_bus_ext1["o2"])


    return c

@gf.cell
def OpticalAligner(                              #optical alignment
        l: float,
        w: float,
        EC_start_position: float = 0,
        EC_length:float = 370,
        EC_N:float = 2
    ): #optical alignment
    c = gf.Component()

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place IOs
    EC_start_position = 0
    EC_length = 370
    EC_N = 2
    EC_1 = c << EC_array(EC_N, EC_pitch)
    EC_1.rotate(270).move((EC_length + 100, EC_start_position))

    # place waveguides
    off = 1500 - l
    c.add_polygon([(off, -w / 2), (off + l, -w / 2), (off + l, w / 2), (off, w / 2)], layer=OXOP)
    c.add_port(name="o1", center=(off, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")
    c.add_port(name="o2", center=(off + l, 0), width=0.5, orientation=0, layer=RIB, port_type="optical")

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    # optical
    routes = []
    routes.append(gf.routing.get_bundle_from_steps(
        c.ports["o1"], EC_1.ports["o1"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        # steps=[{'dx':130+(5-i)*10},{'dy':-10}]
    )[0])
    routes.append(gf.routing.get_bundle_from_steps(
        c.ports["o2"], EC_1.ports["o2"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        steps=[{'dx': 30}, {'dy': -20}]
    )[0])

    for i in routes:
        c.add(i.references)

    # show die
    # c.show()
    return c

@gf.cell
def trombone(deltaL):
  # creates a trombone cell with deltaL length between the two arms
  c=gf.Component()
  p1 = gf.path.straight(length=5)
  p1 += gf.path.euler(radius=15, angle=180, p=0.5, use_eff=False)
  p1 += gf.path.straight(length=5+deltaL/2)
  p1 += gf.path.euler(radius=15, angle=-180, p=0.5, use_eff=False)
  p1 += gf.path.straight(length=5+deltaL/2+5+15)
  p1 += gf.path.euler(radius=15, angle=-90, p=0.5, use_eff=False)
  p1 += gf.path.straight(length=7.849+15+0.698)
  p1 += gf.path.euler(radius=15, angle=90, p=0.5, use_eff=False)
  c.add(gf.path.extrude(p1, cross_section = rib_Oband))
  p2 = gf.path.straight(length=5)
  p2 += gf.path.euler(radius=15, angle=-180, p=0.5, use_eff=False)
  p2 += gf.path.straight(length=5)
  p2 += gf.path.euler(radius=15, angle=180, p=0.5, use_eff=False)
  p2 += gf.path.straight(length=5+5+15)
  p2 += gf.path.euler(radius=15, angle=90, p=0.5, use_eff=False)
  p2 += gf.path.straight(length=7.849+15+0.698)
  p2 += gf.path.euler(radius=15, angle=-90, p=0.5, use_eff=False)
  p2.movey(-10)
  c.add(gf.path.extrude(p2, cross_section = rib_Oband))
  # ports
  c.add_port("o1",center = (0, 0),width = 0.5,orientation = 0,layer = RIB,port_type = "optical")
  c.add_port("o2",center = (24.163+43.325, 0),width = 0.5,orientation = 0,layer = RIB,port_type = "optical")
  c.add_port("o3",center = (0, -10),width = 0.5,orientation = 0,layer = RIB,port_type = "optical")
  c.add_port("o4",center = (24.163+43.325, -10),width = 0.5,orientation = 0,layer = RIB,port_type = "optical")
  #c.pprint_ports()

  return c


@gf.cell
def wgSpool_cell(dop, l, w, xRoute):
    c = gf.Component()

    mySteps = xRoute if xRoute != [] else [{'dx': 20}, {'dy': 80}, {'dx': -100}]
    # determine cell offset from xRoute
    dx, dy = 0, 0
    for el in mySteps:
        dx += el.get('dx', 0)
        dy += el.get('dy', 0)
    dx += 80
    dy -= 80
    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************
    # place IOs
    EC_start_position = 0
    EC_length = 370
    EC_N = 1
    EC_1 = c << EC_array(EC_N, EC_pitch)
    EC_1.rotate(270).move((EC_length + 100, EC_start_position))
    # place a dummy port to terminate the spool
    cc = (dx + 370 - l / 2 + 18, w / 2 + dy + 127 / 2 + 16)
    c.add_port(name="o1", center=cc, width=0.5, orientation=0, layer=RIB, port_type="optical")

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    # optical
    for i in range(22):
        mySteps.extend([{'dy': w - i * 10}, {'dx': -l + i * 10}, {'dy': -(w - 5) + i * 10}, {'dx': l - 5 - i * 10}])
    routes = gf.routing.get_bundle_from_steps(
        EC_1.ports["o1"], c.ports["o1"],
        width=0.41, cross_section=rib_Oband,
        separation=5, radius=15,
        steps=mySteps
    )
    for i in routes:
        c.add(i.references)

    # place doping
    if dop != (0, 0):
        c.add_polygon([(cc[0] - l / 2 - 5, cc[1] - w / 2 - 5), (cc[0] + l / 2 + 5, cc[1] - w / 2 - 5),
                       (cc[0] + l / 2 + 5, cc[1] + w / 2 + 5), (cc[0] - l / 2 - 5, cc[1] + w / 2 + 5)], layer=dop)

    #print(routes[0].length)
    # show die
    # c.show()
    return c


@gf.cell
def EC_TriTip(sec1_len, sec2_len, sec3_len,
              w1, w2, w3, w3_side, w4,
              pitch1, pitch2, pitch3):
    c = gf.Component()

    WG1_cen = c << gf.components.taper(length=sec1_len, width1=w1, width2=w2, layer=RIB)

    xpts = [0, 0, sec1_len, sec1_len]
    ypts = [(pitch1 - w1 / 2), (pitch1 + w1 / 2),
            (pitch2 + w2 / 2), (pitch2 - w2 / 2)]
    c.add_polygon([xpts, ypts], layer=RIB)

    xpts = [0, 0, sec1_len, sec1_len]
    ypts = [(-pitch1 - w1 / 2), (-pitch1 + w1 / 2),
            (-pitch2 + w2 / 2), (-pitch2 - w2 / 2)]
    c.add_polygon([xpts, ypts], layer=RIB)

    WG2_cen = c << gf.components.taper(length=sec2_len, width1=w2, width2=w3, layer=RIB)
    WG2_cen.move((sec1_len, 0))

    xpts = [sec1_len, sec1_len, sec1_len + sec2_len, sec1_len + sec2_len]
    ypts = [(pitch2 - w2 / 2), (pitch2 + w2 / 2),
            (pitch3 + w3_side / 2), (pitch3 - w3_side / 2)]
    c.add_polygon([xpts, ypts], layer=RIB)

    xpts = [sec1_len, sec1_len, sec1_len + sec2_len, sec1_len + sec2_len]
    ypts = [(-pitch2 - w2 / 2), (-pitch2 + w2 / 2),
            (-pitch3 + w3_side / 2), (-pitch3 - w3_side / 2)]
    c.add_polygon([xpts, ypts], layer=RIB)

    WG3_cen = c << gf.components.taper(length=sec3_len, width1=w3, width2=w4, layer=RIB)
    WG3_cen.move((sec1_len + sec2_len, 0))

    c.add_port(
        name="o1",
        center=(sec1_len + sec2_len + sec3_len, 0),
        width=w4,
        orientation=0,
        layer=RIB,
        port_type="optical",
    )

    return c


@gf.cell
def EC_TriTip_loop(sec1_len, sec2_len, sec3_len,
                   w1, w2, w3, w3_side, w4,
                   pitch1, pitch2, pitch3):
    c = gf.Component()

    EC_1 = c << EC_TriTip(sec1_len, sec2_len, sec3_len,
                          w1, w2, w3, w3_side, w4,
                          pitch1, pitch2, pitch3)

    EC_2 = c << EC_TriTip(sec1_len, sec2_len, sec3_len,
                          w1, w2, w3, w3_side, w4,
                          pitch1, pitch2, pitch3)
    EC_2.move((0, 127))

    bend_1 = c << gf.components.bend_euler(radius=15, width=0.41, cross_section=rib_Oband)
    bend_1.connect("o1", EC_1.ports["o1"])

    bend_2 = c << gf.components.bend_euler(radius=15, width=0.41, cross_section=rib_Oband)
    bend_2.connect("o2", EC_2.ports["o1"])

    routes = gf.routing.get_bundle(
        bend_1.ports["o2"],
        bend_2.ports["o1"],
        layer=RIB,
    )
    for route in routes:
        c.add(route.references)

    return c





'''*******************  test functions from old sipho test library *********************************************8'''
#TEST THESE AFTER REMOVING library_sipho import

### Si WG parameters
w_WG = 0.5
Si_WG_pitch = 10

# w_OXOP = 3
# w_IND_WG = 18
# w_si_contact = 7
# gap_si_contact = 3.5

### SI transitions
bend_s_len = 10
slab_taper_len = 5

### directional couplers
WG_elec_connect_len = 12
w_slab_target = 8
MT2_ext_len = 40

### metal
w_MT2 = 20


@gf.cell
def WG_term(length):
    c = gf.Component()

    WG = c << gf.components.taper(width1=0.5, width2=0.2, length=length, layer=RIB)
    doping = c << gf.components.straight(length=length, layer=NCONT, width=4)
    c.add_port("o1", port=WG.ports["o1"])

    return c


@gf.cell
def WG_elec_connect(params:dict):

    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]

    c = gf.Component()

    s1 = gf.Section(width=w_WG, offset=0, layer=RIB, name="central_WG")

    s2 = gf.Section(width=w_slab_target, offset=0, layer=SLAB, name="central_slab")

    offset_si_contact = w_WG / 2 + gap_si_contact + w_si_contact / 2
    s3 = gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact")

    offset_MT1 = w_OXOP / 2 + w_MT1 / 2 + min_gap_OXOP_MT
    s4 = gf.Section(width=w_MT1, offset=offset_MT1, layer=MT1, name="MT1")

    offset_via_1 = w_OXOP / 2 + min_gap_OXOP_MT + min_inc_via_1 + via_size / 2
    s5v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA1), spacing=via_size + gap_via_1, padding=via_size, enclosure=min_inc_via_1, offset=offset_via_1)

    offset_via_2 = w_OXOP / 2 + min_gap_OXOP_MT + min_inc_via_1 + via_size + min_exc_of_via_2 + via_size / 2
    s6v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA2), spacing=min_exc_of_via_2 + via_size, padding=via_size, offset=offset_via_2)

    offset_NCONT = w_slot / 2 + w_slotWG + gap_NCONT_WG + w_NCONT / 2
    s7 = gf.Section(width=w_NCONT + 0.19, offset=offset_NCONT, layer=NCONT, name="NCONT")

    x1 = gf.CrossSection(
        sections=(s1, s2, s3, s4, s7),
        components_along_path=[s5v, s6v],
    )

    p1 = gf.path.straight(length=WG_elec_connect_len)
    _2 = c << gf.path.extrude(p1, x1)

    _1 = c << gf.components.taper_strip_to_ridge(length=slab_taper_len, width1=w_WG, width2=w_WG,
                                                 w_slab1=w_WG, w_slab2=w_slab_target, layer_wg=RIB, layer_slab=SLAB)
    _1.move((-slab_taper_len, 0))

    _3 = c << gf.components.taper_strip_to_ridge(length=slab_taper_len, width1=w_WG, width2=w_WG,
                                                 w_slab1=w_slab_target, w_slab2=w_WG, layer_wg=RIB, layer_slab=SLAB)
    _3.move((WG_elec_connect_len, 0))

    c.add_port("o1", center=(-slab_taper_len, 0), width=w_WG, orientation=180, layer=RIB, port_type="optical")
    c.add_port("o2", center=(WG_elec_connect_len + slab_taper_len, 0), width=w_WG, orientation=0, layer=RIB, port_type="optical")

    return c


@gf.cell
def DC_2X2_FNG_v1(length, params:dict):
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]
    w_IND_WG = params["w_IND_WG"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]
    c = gf.Component()

    w_slot = 0.2

    DC_WG_up = c << gf.components.straight(length=length, width=w_WG, layer=RIB)
    DC_WG_up.move((0, (w_WG + w_slot) / 2))

    DC_WG_low = c << gf.components.straight(length=length, width=w_WG, layer=RIB)
    DC_WG_low.move((0, -(w_WG + w_slot) / 2))

    bend_s = gf.components.bend_s(size=[bend_s_len, Si_WG_pitch / 2 - 0.667], npoints=100, cross_section=rib_Oband)

    bend_1 = c << bend_s
    bend_1.mirror_y()
    bend_1.connect("o1", DC_WG_up.ports["o1"])

    bend_2 = c << bend_s
    bend_2.connect("o1", DC_WG_up.ports["o2"])

    bend_3 = c << bend_s
    bend_3.connect("o1", DC_WG_low.ports["o1"])

    bend_4 = c << bend_s
    bend_4.mirror_y()
    bend_4.connect("o1", DC_WG_low.ports["o2"])

    load_1 = c << WG_elec_connect(params)
    load_1.connect("o1", bend_1.ports["o2"])

    load_2 = c << WG_elec_connect(params)
    load_2.connect("o2", bend_2.ports["o2"])

    load_3 = c << WG_elec_connect(params)
    load_3.connect("o2", bend_3.ports["o2"])

    load_4 = c << WG_elec_connect(params)
    load_4.connect("o1", bend_4.ports["o2"])

    ports_DC_opt = []
    ports = [load_3.ports["o1"],
             load_2.ports["o1"], load_4.ports["o2"], ]
    routes, ports_new = gf.routing.route_ports_to_side(
        ports,
        side="south",
        separation=6,
        radius=15,
        layer=RIB,
        start_straight_length=2,
        extension_length=20,
    )
    for route in routes:
        c.add(route.references)
    ports_DC_opt.extend(ports_new)

    c.add_port("o1", port=ports_DC_opt[0])
    c.add_port("o2", port=ports_DC_opt[1])
    c.add_port("o3", port=ports_DC_opt[2])

    ports_DC_opt_term = []
    ports = [load_1.ports["o2"]]
    routes, ports_new = gf.routing.route_ports_to_side(
        ports,
        side="north",
        separation=6,
        radius=15,
        layer=RIB,
        start_straight_length=2,
        extension_length=5,
    )
    for route in routes:
        c.add(route.references)
    ports_DC_opt_term.extend(ports_new)

    ports = ports_new
    routes, ports_new_2 = gf.routing.route_ports_to_side(
        ports,
        side="east",
        separation=6,
        radius=15,
        layer=RIB,
        start_straight_length=2,
        extension_length=5,
    )
    for route in routes:
        c.add(route.references)
    ports_DC_opt_term.extend(ports_new_2)

    term = c << WG_term(80)
    term.connect("o1", ports_DC_opt_term[1])

    opening_offset = 3.5
    w_OXOP = 3.2
    opening = c << gf.components.straight(length=length + opening_offset * 2, width=w_OXOP, layer=OXOP)
    opening.move((-opening_offset, 0))

    doping_len = length + bend_s_len * 2 + slab_taper_len * 2 + WG_elec_connect_len * 2
    doping = c << gf.components.straight(length=doping_len, width=w_IND_WG, layer=IND)
    doping.move((-doping_len / 2 + length / 2, 0))

    MT2_bridge_len = length + bend_s_len * 2 + slab_taper_len * 4 + WG_elec_connect_len * 2 + MT2_ext_len
    MT2_bridge_up = c << gf.components.straight(length=MT2_bridge_len, width=w_MT2, layer=MT2)
    MT2_bridge_up.move((-MT2_bridge_len / 2 + length / 2, w_MT2))
    MT2_bridge_low = c << gf.components.straight(length=MT2_bridge_len, width=w_MT2, layer=MT2)
    MT2_bridge_low.move((-MT2_bridge_len / 2 + length / 2, -w_MT2))

    c.add_port("e1", center=(-MT2_bridge_len / 2 + length / 2, w_MT2), width=w_MT2,
               orientation=180, layer=MT2, port_type="electrical")
    c.add_port("e2", center=(-MT2_bridge_len / 2 + length / 2, -w_MT2), width=w_MT2,
               orientation=180, layer=MT2, port_type="electrical")
    c.add_port("e3", center=(MT2_bridge_len / 2 + length / 2, w_MT2), width=w_MT2,
               orientation=0, layer=MT2, port_type="electrical")
    c.add_port("e4", center=(MT2_bridge_len / 2 + length / 2, -w_MT2), width=w_MT2,
               orientation=0, layer=MT2, port_type="electrical")

    return c


@gf.cell
def DC_2X2_FNG_v2(length, params:dict):
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]
    c = gf.Component()

    w_slot = 0.2

    DC_WG_up = c << gf.components.straight(length=length, width=w_WG, layer=RIB)
    DC_WG_up.move((0, (w_WG + w_slot) / 2))

    DC_WG_low = c << gf.components.straight(length=length, width=w_WG, layer=RIB)
    DC_WG_low.move((0, -(w_WG + w_slot) / 2))

    bend_s = gf.components.bend_s(size=[bend_s_len, Si_WG_pitch / 2 - 0.667], npoints=100, cross_section=rib_Oband)

    bend_1 = c << bend_s
    bend_1.mirror_y()
    bend_1.connect("o1", DC_WG_up.ports["o1"])

    bend_2 = c << bend_s
    bend_2.connect("o1", DC_WG_up.ports["o2"])

    bend_3 = c << bend_s
    bend_3.connect("o1", DC_WG_low.ports["o1"])

    bend_4 = c << bend_s
    bend_4.mirror_y()
    bend_4.connect("o1", DC_WG_low.ports["o2"])

    load_1 = c << WG_elec_connect(params)
    load_1.connect("o1", bend_1.ports["o2"])

    load_2 = c << WG_elec_connect(params)
    load_2.connect("o2", bend_2.ports["o2"])

    load_3 = c << WG_elec_connect(params)
    load_3.connect("o2", bend_3.ports["o2"])

    load_4 = c << WG_elec_connect(params)
    load_4.connect("o1", bend_4.ports["o2"])

    ports_DC_opt = []
    ports = [load_3.ports["o1"],
             load_2.ports["o1"], load_4.ports["o2"], ]
    routes, ports_new = gf.routing.route_ports_to_side(
        ports,
        side="south",
        separation=6,
        radius=15,
        layer=RIB,
        start_straight_length=2,
        extension_length=20,
    )
    for route in routes:
        c.add(route.references)
    ports_DC_opt.extend(ports_new)

    c.add_port("o1", port=ports_DC_opt[0])
    c.add_port("o2", port=ports_DC_opt[1])
    c.add_port("o3", port=ports_DC_opt[2])

    ports_DC_opt_term = []
    ports = [load_1.ports["o2"]]
    routes, ports_new = gf.routing.route_ports_to_side(
        ports,
        side="north",
        separation=6,
        radius=15,
        layer=RIB,
        start_straight_length=2,
        extension_length=2,
    )
    for route in routes:
        c.add(route.references)
    ports_DC_opt_term.extend(ports_new)

    ports = ports_new
    routes, ports_new_2 = gf.routing.route_ports_to_side(
        ports,
        side="east",
        separation=6,
        radius=15,
        layer=RIB,
        start_straight_length=2,
        extension_length=2,
    )
    for route in routes:
        c.add(route.references)
    ports_DC_opt_term.extend(ports_new_2)

    term = c << WG_term(80)
    term.connect("o1", ports_DC_opt_term[1])

    slab_connect_len = length + bend_s_len * 2 + slab_taper_len * 2
    slab_connect_w = 4
    slab_connect_offset = 8

    s2 = gf.Section(width=slab_connect_w, offset=slab_connect_offset, layer=SLAB, name="slab_1")
    s3 = gf.Section(width=slab_connect_w, offset=-slab_connect_offset, layer=SLAB, name="slab_2")
    s4 = gf.Section(width=slab_connect_w, offset=slab_connect_offset + 2, layer=NCONT, name="NCONT_1")
    s5 = gf.Section(width=slab_connect_w, offset=-slab_connect_offset - 2, layer=NCONT, name="NCONT_2")

    x1 = gf.CrossSection(sections=(s2, s3, s4, s5))
    p1 = gf.path.straight(length=slab_connect_len)
    _2 = c << gf.path.extrude(p1, x1)
    _2.move((-bend_s_len - slab_taper_len, 0))

    slab_align_w = 4.8
    slab_align_offset_x = 3.5
    slab_align_up = c << gf.components.straight(length=length + slab_align_offset_x * 2, width=slab_align_w, layer=SLAB)
    slab_align_up.move((-slab_align_offset_x, slab_align_w / 2 + 1.2))
    slab_align_low = c << gf.components.straight(length=length + 7, width=slab_align_w, layer=SLAB)
    slab_align_low.move((-slab_align_offset_x, -(slab_align_w / 2 + 1.2)))

    opening_offset = 3.5
    w_OXOP = 3.2
    opening = c << gf.components.straight(length=length + opening_offset * 2, width=w_OXOP, layer=OXOP)
    opening.move((-opening_offset, 0))

    doping_len = length + bend_s_len * 2 + slab_taper_len * 2 + WG_elec_connect_len * 2
    doping_offset = 6
    doping_up = c << gf.components.straight(length=doping_len, width=10, layer=IND)
    doping_up.move((-doping_len / 2 + length / 2, doping_offset))
    doping_low = c << gf.components.straight(length=doping_len, width=10, layer=IND)
    doping_low.move((-doping_len / 2 + length / 2, -doping_offset))

    MT2_bridge_len = length + bend_s_len * 2 + slab_taper_len * 4 + WG_elec_connect_len * 2 + MT2_ext_len
    MT2_bridge_up = c << gf.components.straight(length=MT2_bridge_len, width=w_MT2, layer=MT2)
    MT2_bridge_up.move((-MT2_bridge_len / 2 + length / 2, w_MT2))
    MT2_bridge_low = c << gf.components.straight(length=MT2_bridge_len, width=w_MT2, layer=MT2)
    MT2_bridge_low.move((-MT2_bridge_len / 2 + length / 2, -w_MT2))

    c.add_port("e1", center=(-MT2_bridge_len / 2 + length / 2, w_MT2), width=w_MT2,
               orientation=180, layer=MT2, port_type="electrical")
    c.add_port("e2", center=(-MT2_bridge_len / 2 + length / 2, -w_MT2), width=w_MT2,
               orientation=180, layer=MT2, port_type="electrical")
    c.add_port("e3", center=(MT2_bridge_len / 2 + length / 2, w_MT2), width=w_MT2,
               orientation=0, layer=MT2, port_type="electrical")
    c.add_port("e4", center=(MT2_bridge_len / 2 + length / 2, -w_MT2), width=w_MT2,
               orientation=0, layer=MT2, port_type="electrical")

    return c



@gf.cell
def s2s_test_loss(params:dict): #key params of w_slot and w_OXOP
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]

    c = gf.Component()

    len_in = s2s_len_in
    width_in = s2s_width_in
    len_MMI = s2s_len_MMI
    width_MMI = s2s_width_MMI
    len_taper = s2s_len_taper

    xpts = [-len_in, 0, 0, len_MMI, (len_MMI + len_taper),
            (len_MMI + len_taper), len_MMI, len_MMI, (len_MMI + len_taper),
            (len_MMI + len_taper), len_MMI, 0, 0, -len_in]
    ypts = [width_in / 2, width_in / 2, width_MMI / 2, width_MMI / 2, (w_slot / 2 + w_slotWG),
            w_slot / 2, w_slot / 2, -w_slot / 2, -w_slot / 2,
            -(w_slot / 2 + w_slotWG), -width_MMI / 2, -width_MMI / 2, -width_in / 2, -width_in / 2]
    c.add_polygon([xpts, ypts], layer=RIB)

    c.add_port(
        name="o1",
        center=(-len_in, 0),
        width=width_in,
        orientation=180,
        layer=RIB,
        port_type="optical",
    )

    c.add_port(
        name="o2",
        center=((len_MMI + len_taper), 0),
        width=w_slot,
        orientation=0,
        layer=RIB,
        port_type="optical",
    )

    SLAB_min_w = 0.3
    xpts = [len_MMI, (len_MMI + len_taper), (len_MMI + len_taper), len_MMI]
    ypts = [(w_slot / 2 + (width_MMI - w_slot) / 2 - SLAB_min_w), (w_slot / 2 + w_slotWG - buffer_RIB_SLAB_overlay), (w_slot / 2 + w_slotWG + w_slab), (w_slot / 2 + (width_MMI - w_slot) / 2)]
    c.add_polygon([xpts, ypts], layer=SLAB)

    xpts = [len_MMI, (len_MMI + len_taper), (len_MMI + len_taper), len_MMI]
    ypts = [-(w_slot / 2 + (width_MMI - w_slot) / 2 - SLAB_min_w), -(w_slot / 2 + w_slotWG - buffer_RIB_SLAB_overlay), -(w_slot / 2 + w_slotWG + w_slab), -(w_slot / 2 + (width_MMI - w_slot) / 2)]
    c.add_polygon([xpts, ypts], layer=SLAB)

    xpts = [-len_in + 1, (len_MMI + len_taper), (len_MMI + len_taper), -len_in + 1]
    ypts = [w_OXOP / 2, w_OXOP / 2, -w_OXOP / 2, -w_OXOP / 2]
    c.add_polygon([xpts, ypts], layer=OXOP)

    # s17 = gf.Section(width=w_NIM+2, offset=0, layer=NIM, name="NIM")
    # x1 = gf.CrossSection(
    #     sections=[s17]
    #     )
    # p1 = gf.path.straight(length=len_taper+0.1)
    # doping = gf.path.extrude(p1, x1)
    # d = c << doping
    # d.movex(len_MMI-0.1)

    return c


@gf.cell
def PS_slotWG_loss_OXOP(params:dict): #key params of w_slot, w_OXOP, PS_length, gap_IND_WG
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]
    offset_slotWG = (w_slotWG + w_slot) / 2
    s1 = gf.Section(width=w_slotWG, offset=offset_slotWG, layer=RIB, name="slotWG_1")
    s2 = gf.Section(width=w_slotWG, offset=-offset_slotWG, layer=RIB, name="slotWG_2")

    offset_slab = (w_slot + w_slab) / 2 + w_slotWG
    s3 = gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=offset_slab, layer=SLAB, name="slab_1")
    s4 = gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=-offset_slab, layer=SLAB, name="slab_2")

    # offset_si_contact = w_slot/2 + w_slotWG + gap_si_contact + w_si_contact/2
    # s5 = gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact_1")
    # s6 = gf.Section(width=w_si_contact, offset=-offset_si_contact, layer=RIB, name="sl_contact_2")

    # offset_NCONT = w_slot/2 + w_slotWG + gap_NCONT_WG + w_NCONT/2
    # s13 = gf.Section(width=w_NCONT, offset=offset_NCONT, layer=NCONT, name="NCONT_1")
    # s14 = gf.Section(width=w_NCONT, offset=-offset_NCONT, layer=NCONT, name="NCONT_2")

    # offset_IND = w_slot/2 + w_slotWG + gap_IND_WG + w_IND/2
    # s15 = gf.Section(width=w_IND, offset=offset_IND, layer=IND, name="IND_1")
    # s16 = gf.Section(width=w_IND, offset=-offset_IND, layer=IND, name="IND_2")

    # s17 = gf.Section(width=w_NIM, offset=0, layer=NIM, name="NIM")

    s151 = gf.Section(width=w_OXOP, layer=OXOP, name="oxide_open")

    x1 = gf.CrossSection(
        sections=(s1, s2, s3, s4, s151),
        # components_along_path=[s9v, s10v, s11v, s12v],
    )

    p1 = gf.path.straight(length=PS_length)
    PS = gf.path.extrude(p1, x1)

    PS.add_port("o1", center=(0, 0), width=w_slot,
                orientation=180, layer=RIB, port_type="optical")

    PS.add_port("o2", center=(PS_length, 0), width=w_slot,
                orientation=0, layer=RIB, port_type="optical")

    return PS


@gf.cell
def PS_connected_loss_OXOP(params:dict): #key params of w_slot, w_OXOP, PS_length
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]

    c = gf.Component()

    params_noINDgap= {**params, "gap_IND_WG": 0}
    PS = PS_slotWG_loss_OXOP(params_noINDgap)
    ref_PS = c << PS

    s2s_in = s2s_test_loss(params)
    ref_s2s_in = c << s2s_in
    ref_s2s_in.connect("o2", ref_PS.ports["o1"])

    s2s_out = s2s_test_loss(params)
    ref_s2s_out = c << s2s_out
    ref_s2s_out.connect("o2", ref_PS.ports["o2"])

    c.add_port("e_MT2", center=(0, 0), width=1,
               orientation=180, layer=MT2, port_type="electrical")

    c.add_port("o1", port=ref_s2s_in.ports["o1"])
    c.add_port("o2", port=ref_s2s_out.ports["o1"])

    return c


@gf.cell
def PS_slotWG_forSEM(params:dict): #key params w_slot, w_slotWG, w_OXOP, PS_length, gap_IND_WG, gap_NCONT_WG, gap_si_contact, buffer_RIB_SLAB_overlay
    """
    Similar to PS_slotWG_from_params, but has some extra MT2 structures
    """
    w_slot = params["w_slot"]
    w_slotWG = params["w_slotWG"]
    w_slab = params["w_slab"]
    PS_length = params["PS_length"]
    buffer_RIB_SLAB_overlay = params["buffer_RIB_SLAB_overlay"]

    w_OXOP = params["w_OXOP"]
    w_si_contact = params["w_si_contact"]
    gap_si_contact = params["gap_si_contact"]

    w_NCONT = params["w_NCONT"]
    gap_NCONT_WG = params["gap_NCONT_WG"]
    w_IND = params["w_IND"]
    gap_IND_WG = params["gap_IND_WG"]
    w_NIM = params["w_NIM"]

    w_MT1 = params["w_MT1"]
    min_gap_OXOP_MT = params["min_gap_OXOP_MT"]
    via_size = params["via_size"]
    gap_via_1 = params["gap_via_1"]
    min_inc_via_1 = params["min_inc_via_1"]
    min_exc_of_via_2 = params["min_exc_of_via_2"]

    offset_slotWG = (w_slotWG + w_slot) / 2
    s1 = gf.Section(width=w_slotWG, offset=offset_slotWG, layer=RIB, name="slotWG_1")
    s2 = gf.Section(width=w_slotWG, offset=-offset_slotWG, layer=RIB, name="slotWG_2")

    offset_slab = (w_slot + w_slab) / 2 + w_slotWG
    s3 = gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=offset_slab, layer=SLAB, name="slab_1")
    s4 = gf.Section(width=(w_slab + buffer_RIB_SLAB_overlay * 2), offset=-offset_slab, layer=SLAB, name="slab_2")

    offset_si_contact = w_slot / 2 + w_slotWG + gap_si_contact + w_si_contact / 2
    s5 = gf.Section(width=w_si_contact, offset=offset_si_contact, layer=RIB, name="si_contact_1")
    s6 = gf.Section(width=w_si_contact, offset=-offset_si_contact, layer=RIB, name="sl_contact_2")

    offset_MT1 = w_OXOP / 2 + w_MT1 / 2 + min_gap_OXOP_MT
    s7 = gf.Section(width=w_MT1, offset=offset_MT1, layer=MT1, name="MT1_1")
    s8 = gf.Section(width=w_MT1, offset=-offset_MT1, layer=MT1, name="MT1_2")

    offset_MT2 = w_OXOP / 2 + w_MT1 / 2 + min_gap_OXOP_MT
    s72 = gf.Section(width=w_MT1, offset=offset_MT1, layer=MT2, name="MT2_1")
    s82 = gf.Section(width=w_MT1, offset=-offset_MT1, layer=MT2, name="MT2_2")

    offset_via_1 = w_OXOP / 2 + min_gap_OXOP_MT + min_inc_via_1 + via_size / 2
    s9v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA1), spacing=via_size + gap_via_1, padding=via_size, enclosure=min_inc_via_1, offset=offset_via_1)
    s10v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA1), spacing=via_size + gap_via_1, padding=via_size, enclosure=min_inc_via_1, offset=-offset_via_1)

    offset_via_2 = w_OXOP / 2 + min_gap_OXOP_MT + min_inc_via_1 + via_size + min_exc_of_via_2 + via_size / 2
    s11v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA2), spacing=min_exc_of_via_2 + via_size, padding=via_size, offset=offset_via_2)
    s12v = ComponentAlongPath(component=gf.c.via(size=(3, 3), layer=VIA2), spacing=min_exc_of_via_2 + via_size, padding=via_size, offset=-offset_via_2)

    offset_NCONT = w_slot / 2 + w_slotWG + gap_NCONT_WG + w_NCONT / 2
    s13 = gf.Section(width=w_NCONT, offset=offset_NCONT, layer=NCONT, name="NCONT_1")
    s14 = gf.Section(width=w_NCONT, offset=-offset_NCONT, layer=NCONT, name="NCONT_2")

    offset_IND = w_slot / 2 + w_slotWG + gap_IND_WG + w_IND / 2
    s15 = gf.Section(width=w_IND, offset=offset_IND, layer=IND, name="IND_1")
    s16 = gf.Section(width=w_IND, offset=-offset_IND, layer=IND, name="IND_2")

    s17 = gf.Section(width=w_NIM, offset=0, layer=NIM, name="NIM")

    s151 = gf.Section(width=w_OXOP, layer=OXOP, name="oxide_open")

    x1 = gf.CrossSection(
        sections=(s1, s2, s3, s4, s5, s6, s7, s8, s72, s82, s13, s14, s15, s16, s17, s151),
        components_along_path=[s9v, s10v, s11v, s12v],
    )

    p1 = gf.path.straight(length=PS_length)
    PS = gf.path.extrude(p1, x1)

    # Clear existing ports and add custom ports
    PS.ports.clear()

    PS.add_port(
        name="o1",
        center=(0, 0),
        width=w_slot,
        orientation=180,
        layer=RIB,
        port_type="optical",
    )

    PS.add_port(
        name="o2",
        center=(PS_length, 0),
        width=w_slot,
        orientation=0,
        layer=RIB,
        port_type="optical",
    )

    return PS


@gf.cell
def SEM_slotWG(params:dict): #key params w_slot, w_slotWG, w_OXOP, PS_length, gap_IND_WG, gap_NCONT_WG, gap_si_contact, buffer_RIB_SLAB_overlay
    c = gf.Component()

    PS = PS_slotWG_forSEM(params)#w_slot, w_slotWG, w_OXOP, PS_length, gap_IND_WG,
                          #gap_NCONT_WG, gap_si_contact, buffer_RIB_SLAB_overlay)
    ref_PS = c << PS

    return c


@gf.cell
def SEM_160nm_2um(text_layer, params:dict):
    c = gf.Component()

    params={**params, "w_slotWG": 0.3, "w_OXOP":2, "PS_length": 180, "gap_IND_WG":0.2,
            "gap_NCONT_WG":2.5, "gap_si_contact":3.5, "buffer_RIB_SLAB_overlay": 0.07}

    for i in range(6):
        params={**params, "w_slot": 0.15 + 0.004 * i}
        _ = c << SEM_slotWG(params) #fix later
        _.rotate(90).move((33 * i, 0))

    labeling_SEM = c << gf.components.text(text='SEM1', size=20, position=(86, 190), justify='center', layer=text_layer)

    return c


@gf.cell
def SEM_200nm_2um(text_layer, params:dict):
    c = gf.Component()
    params = {**params, "w_slotWG": 0.3, "w_OXOP": 2, "PS_length": 180, "gap_IND_WG": 0.2,
              "gap_NCONT_WG": 2.5, "gap_si_contact": 3.5, "buffer_RIB_SLAB_overlay": 0.07}
    for i in range(6):
        params = {**params, "w_slot": 0.19 + 0.004 * i}
        _ = c << SEM_slotWG(params)
        _.rotate(90).move((33 * i, 0))

    labeling_SEM = c << gf.components.text(text='SEM2', size=20, position=(86, 190), justify='center', layer=text_layer)

    return c


@gf.cell
def SEM_160nm_3um(text_layer, params:dict):
    c = gf.Component()
    params = {**params, "w_slotWG": 0.3, "w_OXOP": 3, "PS_length": 180, "gap_IND_WG": 0.2,
              "gap_NCONT_WG": 2.5, "gap_si_contact": 3.5, "buffer_RIB_SLAB_overlay": 0.07}
    for i in range(6):
        params = {**params, "w_slot": 0.15 + 0.004 * i}
        _ = c << SEM_slotWG(params)
        _.rotate(90).move((34 * i, 0))

    labeling_SEM = c << gf.components.text(text='SEM3', size=20, position=(86, 190), justify='center', layer=text_layer)

    return c


@gf.cell
def SEM_200nm_3um(text_layer, params:dict):
    c = gf.Component()
    params = {**params, "w_slotWG": 0.3, "w_OXOP": 3, "PS_length": 180, "gap_IND_WG": 0.2,
              "gap_NCONT_WG": 2.5, "gap_si_contact": 3.5, "buffer_RIB_SLAB_overlay": 0.07}
    for i in range(6):
        params = {**params, "w_slot": 0.19 + 0.004 * i}
        _ = c << SEM_slotWG(params)
        _.rotate(90).move((34 * i, 0))

    labeling_SEM = c << gf.components.text(text='SEM4', size=20, position=(86, 190), justify='center', layer=text_layer)

    return c


@gf.cell
def SEM_160nm_3um_WG_w(text_layer, params:dict):
    c = gf.Component()
    params = {**params, "w_slot": 0.16, "w_OXOP": 3, "PS_length": 180, "gap_IND_WG": 0.2,
              "gap_NCONT_WG": 2.5, "gap_si_contact": 3.5, "buffer_RIB_SLAB_overlay": 0.07}
    for i in range(6):
        params = {**params, "w_slotWG": 0.3 - 0.02 * i}
        _ = c << SEM_slotWG(params)
        _.rotate(90).move((34 * i, 0))

    labeling_SEM = c << gf.components.text(text='SEM5', size=20, position=(86, 190), justify='center', layer=text_layer)

    return c


##### from PDK file

@gf.cell 
def GC_array_v1(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        print("GC_TE")
        ref_GC = c << GC_TE
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    length=2
    width=0.5
    _1 = c << gf.components.straight(length=length, cross_section = rib_Oband)
    _1.connect('o1', c.ports["o1"])
    _2 = c << gf.components.bend_euler(radius=15, angle=90, npoints=40, cross_section = rib_Oband)
    _2.connect('o2', _1.ports["o2"])
   
    _3 = c << gf.components.straight(length=length,cross_section = rib_Oband)
    _3.connect('o1', c.ports["o2"])
    _4 = c << gf.components.bend_euler(radius=15, angle=90, npoints=40, cross_section = rib_Oband)
    _4.connect('o1', _3.ports["o2"])   

    routes = gf.routing.get_bundle( _4.ports['o2'], _2.ports['o1'],cross_section = rib_Oband)
    c.add(routes[0].references)

    return c

    



#@gf.cell 
def GC_array_v2(N, pitch, rot):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC_TE
        bb_ = ref_GC.bbox
        ref_GC.rotate(rot).movex(-bb_[0,0]). movey(-bb_[1,1] - i*pitch )
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    # length=2
    # width=0.5
    # _1 = c << gf.components.straight(length=length, width=width, cross_section = rib_Oband)
    # _1.connect('o1', c.ports[f"o{N+1}"] )
    # _2 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, width=width, cross_section = rib_Oband)
    # _2.connect('o2', _1.ports["o2"] )
   
    # _3 = c << gf.components.straight(length=length, width=width, cross_section = rib_Oband)
    # _3.connect('o1', c.ports[f"o{N+2}"] )
    # _4 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, width=width, cross_section = rib_Oband)
    # _4.connect('o1', _3.ports["o2"] )   

    # routes = gf.routing.get_bundle( _4.ports['o2'], _2.ports['o1'], cross_section = rib_Oband)
    # c.add(routes[0].references)

    return c   
