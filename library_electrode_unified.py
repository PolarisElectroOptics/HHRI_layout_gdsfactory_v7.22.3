import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection, cross_section, ComponentAlongPath
from functools import partial
import math
#from shapely.geometry.polygon import Polygon
import shapely
from shapely.geometry.polygon import Polygon


from layer_info import *
#from library_sipho import *
from library_electrode_params import *

'''Table of Contents:
- GSGSG structures
- DC Pads
- other misc. structures
'''


def PEO_cheese(poly, slot_size, gap_slot, border_slot, layer_cheese, rotation_degrees=0):
    """
    returns a layer of cheesing shapes for a given poly
    :param poly: list of points defining shape to be cheesed, ctrclkwise from bottom left: [bottom_left, bottom_right, top_right, top_left]
    :param slot_size: tuple
    :param gap_slot: float, space required between slots
    :param rotation_degrees: angle by which cheese shapes will be rotated
    :param layer_cheese: tuple, GDS layer of cheese slots to be placed
    :return: gf.Component

    example:
        bl = (pad_length, -pad_center_gnd_width/2)
        br = (pad_length + trans_length, -S2S_center_gnd_width/2)
        tr = (pad_length + trans_length, S2S_center_gnd_width/2)
        tl = (pad_length, pad_center_gnd_width/2)
        poly = [bl, br, tr, tl]
        slot_size = (10, 4)
        gap_slot = 10
        layer_slot = MT3_SLOT

        c << PEO_cheese(poly, slot_size, gap_slot, layer_slot)
    """
    c = gf.Component("cheese")
    poly_MT = Polygon(poly)
    poly_MT_buff = poly_MT.buffer(-border_slot)

    bounds_poly_MT_buff = poly_MT_buff.bounds  # (minx, miny, maxx, maxy)
    cheese_area_width = bounds_poly_MT_buff[2] - bounds_poly_MT_buff[0]
    cheese_area_height = bounds_poly_MT_buff[3] - bounds_poly_MT_buff[1]

    # generate array, only place component reference if it will be withing buffered bounding box
    c_slot = gf.components.rectangle(size=slot_size, layer=layer_cheese)
    delta_x = math.sqrt( gap_slot**2 - (gap_slot/2 - slot_size[1]/2)**2 )

    # print("array start:")
    # print((poly[0][0], poly[0][1]))
    for i in range(int(2*int(cheese_area_width / (slot_size[0] + gap_slot)))):# + gap_slot))): #col
        if i % 2 == 1:  # odd
            place_point_y_offset = (gap_slot+ slot_size[1])/4
        else:
            place_point_y_offset = -(gap_slot+ slot_size[1])/4
        for j in range(int(2*(cheese_area_height / (slot_size[1] + gap_slot)))):# + gap_slot))): #row
            place_point = (-cheese_area_width/5 + poly[0][0] + i * (slot_size[0] + delta_x),   poly[0][1] + j*(slot_size[1] + gap_slot) + place_point_y_offset)
            place_point_x_rot = place_point[0]*math.cos(math.radians(rotation_degrees)) - place_point[1]*math.sin(math.radians(rotation_degrees))
            place_point_y_rot = place_point[0]*math.sin(math.radians(rotation_degrees)) + place_point[1]*math.cos(math.radians(rotation_degrees))
            place_point = (place_point_x_rot, place_point_y_rot)
            poly_slot = shapelyP.Polygon([(place_point[0], place_point[1]),
                                          (place_point[0] + slot_size[0], place_point[1]),
                                          (place_point[0] + slot_size[0], place_point[1] + slot_size[1]),
                                          (place_point[0], place_point[1] + slot_size[1])])
            poly_slot = shapely.affinity.rotate(poly_slot, angle=rotation_degrees)
            if shapely.contains(poly_MT_buff, poly_slot):  # if slot will be within buffered electrode shape
                _ = c << c_slot
                _.rotate(rotation_degrees)
                _.move(place_point)

    return c


def PEO_custom_via_from_line(
        line_coor_x = [0, 50],  # line coordinates [[x0, x1], [y0,y1]]
        line_coor_y = [0, 50],
        via_n=10,  # no of via per column
        via_space=0.45,  # pitch of via
        via_width=0.35,  # via width
        via_vdistfromline=5,  # via center distance from line [to_upward, to_downward]
        via_layer=VIA1,  # via layer
):
    line_coor = list(zip(line_coor_x, line_coor_y))
    via_sep = via_width + via_space
    total_height = via_sep * (via_n-1) + via_width/2
    m = (line_coor[1][1] - line_coor[0][1]) / (line_coor[1][0] - line_coor[0][0])
    x = line_coor[0][0]

    c = gf.Component("Vias")

    via_ = gf.components.rectangle(size=(via_width, via_width), layer=via_layer, centered=True, port_type='electrical', port_orientations=(180, 90, 0, -90))
    via_a = []
    while (x <= line_coor[1][0]):

        y_ = line_coor[0][1] + m * (x - line_coor[0][0])

        if via_vdistfromline > 0:
            via_offset = via_vdistfromline + via_width/2
        else:
            via_offset = -total_height + via_vdistfromline

        # finding y centers of via
        if via_n:
            for ii in range(1, via_n + 1):
                y = y_ + via_offset + (ii - 1) * via_sep
                # print((x,y))
                via_a.append(c << via_)
                via_a[-1].movex(x).movey(y)
        x = x + via_sep

    return c


# def GSG_piece_old(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, pad_length, layer_MT2): #obsolete
#     p = gf.path.straight(length=pad_length, npoints=2)
#
#     off_s1 = pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width/2
#     s1 = gf.Section(width=pad_sig_width, offset=off_s1, layer=layer_MT2, name="sig1")
#
#     s0 = gf.Section(width=pad_center_gnd_width, offset=0, layer=layer_MT2, name="center gnd", port_names=("e1", "e2"), port_types=("electrical", "electrical"))
#
#     off_s2 = pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width + pad_outer_gnd_width/2
#     s2 = gf.Section(width=pad_outer_gnd_width, offset=off_s2, layer=layer_MT2, name="outer gnd1")
#
#     x1 = gf.CrossSection(sections=(s0,s1,s2))
#     generate = gf.path.extrude(p, x1)
#     return generate, x1


def GSG_piece(params, pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, pad_length, layer_MTX, slot=False, layer_MTX_SLOT=MT3_SLOT, num_rows_slot = 1 ):
    cheese_slot_size_MT3 = 0#params["cheese_slot_size_MT3"]
    gap_slot_MT3 = 0# params["gap_slot_MT3"]
    border_slot_MT3 = 0# params["border_slot_MT3"]

    components_along_path = []
    p = gf.path.straight(length=pad_length, npoints=2)

    off_s1 = -(pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2)
    s1 = gf.Section(width=pad_sig_width, offset=off_s1, layer=layer_MTX, name="sig1")
    if slot and pad_sig_width > 30:
        spacing = min(pad_sig_width/num_rows_slot, 30)
        for i in range(num_rows_slot):
            row_offset_factor = 0.5*num_rows_slot + 0.5
            components_along_path.append(ComponentAlongPath(component=gf.c.via(size=cheese_slot_size_MT3, layer=layer_MTX_SLOT), spacing=spacing, padding=border_slot_MT3, offset= (-(num_rows_slot-row_offset_factor)*(spacing+ cheese_slot_size_MT3[0]) + i*(spacing+cheese_slot_size_MT3[0]) + off_s1 )))
            components_along_path.append(ComponentAlongPath(component=gf.c.via(size=cheese_slot_size_MT3, layer=layer_MTX_SLOT), spacing=spacing, padding=border_slot_MT3, offset=-(-(num_rows_slot-row_offset_factor)*(spacing+ cheese_slot_size_MT3[0]) + i*(spacing+cheese_slot_size_MT3[0]) + off_s1 )))

    s0 = gf.Section(width=pad_center_gnd_width, offset=0, layer=layer_MTX, name="center gnd", port_names=("e1", "e2"), port_types=("electrical", "electrical"))
    if slot and pad_center_gnd_width > 30:
        for i in range(num_rows_slot):
            components_along_path.append(ComponentAlongPath(component=gf.c.via(size=cheese_slot_size_MT3, layer=layer_MTX_SLOT), spacing=spacing, padding=border_slot_MT3, offset=-(num_rows_slot-row_offset_factor)*(spacing+ cheese_slot_size_MT3[0]) + i*(spacing+cheese_slot_size_MT3[0])))

    off_s2 = -off_s1
    s2 = gf.Section(width=pad_sig_width, offset=off_s2, layer=layer_MTX, name="sig2")

    x1 = gf.CrossSection(sections=(s0,s1,s2), components_along_path=components_along_path)
    generate = gf.path.extrude(p, x1)
    return generate, x1   



def SGS_piece(
    pad_center_gnd_width: float,
    pad_inner_gap_width: float,
    pad_sig_width: float,
    pad_outer_gap_width: float,
    pad_outer_gnd_width: float,
    pad_length: float,
    layer_MT2,  # Can be int or tuple
) -> tuple[Component, CrossSection]:
    """
    Creates a symmetric GSGSG (Ground-Signal-Ground-Signal-Ground) electrical cross-section structure.

    Args:
        pad_center_gnd_width: Width of the central ground trace.
        pad_inner_gap_width: Gap between center ground and signal traces.
        pad_sig_width: Width of signal traces (both top and bottom).
        pad_outer_gap_width: Gap between signal and outer ground traces.
        pad_outer_gnd_width: Width of outer ground traces.
        pad_length: Length of the extruded section.
        layer_MT2: Layer specification (int or tuple).

    Returns:
        A tuple containing:
        - The extruded GSGSG component
        - The associated CrossSection object
    """
    # Handle both integer and tuple layer formats
    layer_spec = (layer_MT2, 0) if isinstance(layer_MT2, int) else layer_MT2
    
    # Create straight path
    p = gf.path.straight(length=pad_length, npoints=2)

    # Define trace sections
    s0 = gf.Section(
        width=pad_center_gnd_width,
        offset=0,
        layer=layer_spec,
        name="center gnd",
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical")
    )

    off_s1 = pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2
    s1 = gf.Section(width=pad_sig_width, offset=off_s1, layer=layer_spec, name="sig1")
    s11 = gf.Section(width=pad_sig_width, offset=-off_s1, layer=layer_spec, name="sig2")

    # Build cross-section and extrude
    x1 = gf.CrossSection(sections=(s0, s1, s11))
    comp = gf.path.extrude(p, x1)

    return comp, x1



@gf.cell 
def SGS_MT2_DC(PS_length, trans_length, taper_type, sig_trace, params : dict, config="standard",):
    """
    Unified GSGSG MT2 DC function combining all variants.
    
    Args:
        PS_length: Phase shifter length
        trans_length: Transition length  
        taper_type: 1 (PS params) or 2 (S2S params)
        sig_trace: "narrow", "medium", or "wide"
        config: "standard", "batch", or "compact"
    """

    PS_center_gnd_width = params["PS_center_gnd_width"] #we should just call the params directly when needed..
    PS_inner_gap_width = params["PS_inner_gap_width"]
    PS_sig_width = params["PS_sig_width"]
    PS_outer_gap_width = params["PS_outer_gap_width"]
    PS_outer_gnd_width = params["PS_outer_gnd_width"]

    S2S_center_gnd_width = params["S2S_center_gnd_width"]
    S2S_inner_gap_width = params["S2S_inner_gap_width"]
    S2S_sig_width = params["S2S_sig_width"]
    S2S_outer_gap_width = params["S2S_outer_gap_width"]
    S2S_outer_gnd_width = params["S2S_outer_gnd_width"]
    S2S_length = params["S2S_length"]

    pad_center_gnd_width = params["pad_center_gnd_width"]
    pad_inner_gap_width = params["pad_inner_gap_width"]
    pad_sig_width = params["pad_sig_width"]
    pad_outer_gap_width = params["pad_outer_gap_width"]
    pad_outer_gnd_width = params["pad_outer_gnd_width"]
    pad_length = params["pad_length"]

    sc_length = params["sc_length"]
    layer_MT2 = params["layer_MT2"]
    layer_PAD = params["layer_PAD"]
    spacing_x = params["spacing_x"]
    spacing_y = params["spacing_y"]
    DC_pad_size_x = params["DC_pad_size_x"]
    DC_pad_size_y = params["DC_pad_size_y"]
    pads_rec_gap = params["pads_rec_gap"]

    c = gf.Component()
    # Define PS parameters based on config
    if config == "compact":
        PS_center_gnd_width_local = 170
        PS_inner_gap_width_local = 13
        PS_sig_width_base = 10
        PS_outer_gap_width_base = 34
        PS_outer_gnd_width_local = 60
        
        S2S_center_gnd_width_base = 170
        S2S_inner_gap_width_base = 13
        S2S_sig_width_base = 34
        S2S_outer_gap_width_base = 10
        S2S_outer_gnd_width_base = 140
    elif config == "standard" or config == "batch":  # "standard" or "batch"
        PS_center_gnd_width_local = PS_center_gnd_width
        PS_inner_gap_width_local = PS_inner_gap_width
        PS_sig_width_base = PS_sig_width
        PS_outer_gap_width_base = PS_outer_gap_width
        PS_outer_gnd_width_local = PS_outer_gnd_width
        
        S2S_center_gnd_width_base = S2S_center_gnd_width
        S2S_inner_gap_width_base = S2S_inner_gap_width
        S2S_sig_width_base = S2S_sig_width
        S2S_outer_gap_width_base = S2S_outer_gap_width
        S2S_outer_gnd_width_base = S2S_outer_gnd_width

    # Define PS signal trace parameters
    if sig_trace == "narrow":
        PS_sig_width_local = PS_sig_width_base
        PS_outer_gap_width_local = PS_outer_gap_width_base
    elif sig_trace in {"medium", "wide"}:
        PS_sig_width_local = 60
        PS_outer_gap_width_local = 90       
    else: 
        raise ValueError(f"Invalid sig_trace: {sig_trace}")

    # Define S2S taper parameters
    if taper_type == 1:   
        S2S_center_gnd_width_local = PS_center_gnd_width_local
        S2S_inner_gap_width_local = PS_inner_gap_width_local
        S2S_sig_width_local = PS_sig_width_local
        S2S_outer_gap_width_local = PS_outer_gap_width_local
        S2S_outer_gnd_width_local = PS_outer_gnd_width_local
    elif taper_type == 2:
        S2S_center_gnd_width_local = S2S_center_gnd_width_base
        S2S_inner_gap_width_local = S2S_inner_gap_width_base
        S2S_sig_width_local = S2S_sig_width_base
        S2S_outer_gap_width_local = S2S_outer_gap_width_base
        S2S_outer_gnd_width_local = S2S_outer_gnd_width_base
    else:
        raise ValueError(f"Invalid taper_type: {taper_type}")

    # Build components (common structure)
    # Input contact pads
    pad_in, x1 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, pad_length, layer_MT2)
    _ = c << pad_in
    pad_in_PAD, _ = SGS_piece(pad_center_gnd_width-10, pad_inner_gap_width+10, pad_sig_width-10, pad_outer_gap_width+10, pad_outer_gnd_width-10, pad_length-10, layer_PAD)
    _ = c << pad_in_PAD
    _.movex(5)

    # S2S 1
    s2s_1, x2 = SGS_piece(S2S_center_gnd_width_local, S2S_inner_gap_width_local, S2S_sig_width_local, S2S_outer_gap_width_local, S2S_outer_gnd_width_local, S2S_length, layer_MT2)
    _ = c << s2s_1
    _.movex(trans_length + pad_length)

    # Phase shifter region
    PS, x3 = SGS_piece(PS_center_gnd_width_local, PS_inner_gap_width_local, PS_sig_width_local, PS_outer_gap_width_local, PS_outer_gnd_width_local, PS_length, layer_MT2)
    _ = c << PS
    _.movex(trans_length + pad_length + S2S_length)

    # Transition 1 
    taper1 = gf.components.taper_cross_section_linear(x1, x2, length=trans_length)
    _ = c << taper1
    _.movex(pad_length)
    
    # S2S 2
    s2s_2, x4 = SGS_piece(S2S_center_gnd_width_local, S2S_inner_gap_width_local, S2S_sig_width_local, S2S_outer_gap_width_local, S2S_outer_gnd_width_local, S2S_length, layer_MT2)
    _ = c << s2s_2
    _.movex(trans_length + pad_length + S2S_length + PS_length)

    # Different endings based on config
    if config == "compact":
        # Through connection with output pads
        pad_out, x5 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, pad_length, layer_MT2)
        _ = c << pad_out
        _.movex(trans_length*2 + pad_length + S2S_length*2 + PS_length)
        pad_out_PAD, _ = SGS_piece(pad_center_gnd_width-10, pad_inner_gap_width+10, pad_sig_width-10, pad_outer_gap_width+10, pad_outer_gnd_width-10, pad_length-10, layer_PAD)
        _ = c << pad_out_PAD
        _.movex(5 + trans_length*2 + pad_length + S2S_length*2 + PS_length)
        
        # Transition 2
        taper2 = gf.components.taper_cross_section_linear(x4, x5, length=trans_length)
        _ = c << taper2
        _.movex(pad_length + trans_length + PS_length + S2S_length*2)
    else:
        # Short circuit (standard and batch)
        g_extend, _ = SGS_piece(S2S_center_gnd_width_local, S2S_inner_gap_width_local +S2S_sig_width_local, 0, S2S_outer_gap_width_local, S2S_outer_gnd_width_local, sc_length, layer_MT2)
        _ = c << g_extend
        _.movex(trans_length + pad_length + S2S_length + PS_length + S2S_length)
        g_connect, _ = SGS_piece(S2S_center_gnd_width_local, 0, S2S_inner_gap_width_local +S2S_sig_width_local +S2S_outer_gap_width_local, 0, S2S_outer_gnd_width_local, sc_length, layer_MT2)
        _ = c << g_connect
        _.movex(trans_length + pad_length + S2S_length + PS_length + S2S_length + sc_length)

    # Batch-specific extensions
    if config == "batch":
        extend_len = 10
        extend_1_MT2 = c << gf.components.rectangle(size=(extend_len, pad_sig_width), layer=MT2)
        extend_1_MT2.move((-extend_len, -(pad_center_gnd_width)/2 -pad_sig_width -pad_inner_gap_width))
        via_1 = c << gf.components.rectangle(size=(extend_len/2, pad_sig_width-10), layer=VIA2)
        via_1.move((-extend_len*3/4, -(pad_center_gnd_width)/2 -pad_sig_width -pad_inner_gap_width+5))
        extend_1_MT1 = c << gf.components.rectangle(size=(extend_len, pad_sig_width), layer=MT1)
        extend_1_MT1.move((-extend_len, -(pad_center_gnd_width)/2 -pad_sig_width -pad_inner_gap_width))
        
        extend_2_MT2 = c << gf.components.rectangle(size=(extend_len*3, pad_sig_width), layer=MT2)
        extend_2_MT2.move((-extend_len*3, (pad_center_gnd_width)/2 +pad_inner_gap_width))

    # Add ports
    c.add_port(
        name="e_low",
        center=(pad_length + trans_length + S2S_length, 
                -PS_center_gnd_width_local/2 - PS_inner_gap_width_local/2),
        width=1,
        orientation=0, 
        layer=MT2,
        port_type="electrical",
    )
 
    c.add_port(
        name="e_up",
        center=(pad_length + trans_length + S2S_length, 
                PS_center_gnd_width_local/2 + PS_inner_gap_width_local/2),
        width=1,
        orientation=0, 
        layer=MT2,
        port_type="electrical",
    )
    
    return c






# @gf.cell #unused for SilTerra
# def GSG_MTX(params:dict, taper_type=2, sig_trace="narrow", termination=0, layer_MTX=MT2) -> Component:
#
#    #  if "taper_type" in params:  taper_type = params["taper_type"]
#    #  else: taper_type = 2
#    #  if "sig_trace" in params:   sig_trace = params["sig_trace"]
#    #  else: sig_trace = "narrow"
#    #  if "termination" in params: termination = params["termination"]
#    #  else: termination = 0
#
#     layer_PAD = params["layer_PAD"]
#     layer_HTR = params["layer_HTR"]
#     layer_VIA2 = params["layer_VIA2"]
#     trans_length = params["trans_length"]
#     PS_length = params["PS_length"]
#     pad_length = params["pad_length"]
#     pad_center_gnd_width = params["pad_center_gnd_width"]
#     pad_inner_gap_width = params["pad_inner_gap_width"]
#     pad_sig_width = params["pad_sig_width"]
#     PS_center_gnd_width = params["PS_center_gnd_width"]
#     PS_inner_gap_width = params["PS_inner_gap_width"]
#     PS_sig_width = params["PS_sig_width"]
#
#     PS_MT1_center_gnd_width = params["PS_MT1_center_gnd_width"]
#     PS_MT1_inner_gap_width = params["PS_MT1_inner_gap_width"]
#     PS_MT1_sig_width = params["PS_MT1_sig_width"]
#
#     S2S_center_gnd_width = params["S2S_center_gnd_width"]
#     S2S_inner_gap_width = params["S2S_inner_gap_width"]
#     S2S_sig_width = params["S2S_sig_width"]
#     S2S_length = params["S2S_length"]
#     cheese = params["cheese"]
#     cheese_slot_size_MT3 = params["cheese_slot_size_MT3"]
#     gap_slot_MT3 = params["gap_slot_MT3"]
#     border_slot_MT3 = params["border_slot_MT3"]
#     center_slot_line_offset_MT3 = params["center_slot_line_offset_MT3"]
#     cheese_shift_x = params["cheese_shift_x"]
#     cheese_shift_y = params["cheese_shift_y"]
#
#     center_slot_line_length_delta_MT3_term = params["center_slot_line_length_delta_MT3_term"]
#
#     cheese_slot_size_MT1 = params["cheese_slot_size_MT1"]
#
#     via_size_top = params["via_size_top"]
#     gap_via_top = params["gap_via_top"]
#     via_size_1 = params["via_size_1"]
#     gap_via_1 = params["gap_via_1"]
#     via_size_contact = params["via_size_contact"][0]
#     gap_via_contact = params["gap_via_contact"]
#     num_rows_V1 = params["num_rows_V1"]
#
#     MT_dummy_margin = params["MT_dummy_margin"]
#
#
#
#     c = gf.Component()
#
#     '''*******************************************setting and evaluating parameter presets ***************************************************'''
#
#     #phase shifter parameters
#     if sig_trace == "narrow":
#         pass
#     elif sig_trace == "medium":
#         params["PS_sig_width"] = 60
#         params["PS_outer_gap_width"] = 90
#         # print('medium')
#     elif sig_trace == "wide":
#         params["PS_sig_width"] = 60
#         params["PS_outer_gap_width"] = 90
#
#     # elif sig_trace == "test_GSG_1":  # 44.8 Ohm
#     #     ps_params["pad_sig_width"] = 200
#     #     ps_params["pad_outer_gap_width"] = 150
#
#     # elif sig_trace == "test_GSG_2":  # 50.2 Ohm
#     #     ps_params["pad_sig_width"] = 145
#     #     ps_params["pad_outer_gap_width"] = 20
#     #
#     # elif sig_trace == "test_GSG_3":  # 54.6 Ohm
#     #     ps_params["pad_sig_width"] = 70
#     #     ps_params["pad_outer_gap_width"] = 80
#     #
#     # elif sig_trace == "test_GSG_4":  # 61.3 Ohm
#     #     ps_params["pad_sig_width"] = 30
#     #     ps_params["pad_outer_gap_width"] = 120
#     else:
#         raise ValueError("check PS MT2 sig_trace")
#
#     #else leave as default params
#
#     ### taper type == pads to PS
#     if taper_type == 1:
#         params["S2S_center_gnd_width"] = params["PS_center_gnd_width"]
#         params["S2S_inner_gap_width"] = params["PS_inner_gap_width"]
#         params["S2S_sig_width"] = params["PS_sig_width"]
#         params["S2S_outer_gap_width"] = params["PS_outer_gap_width"]
#         params["S2S_outer_gnd_width"] = params["PS_outer_gnd_width"]
#         S2S_center_gnd_width = params["PS_center_gnd_width"]
#         S2S_inner_gap_width = params["PS_inner_gap_width"]
#         S2S_sig_width = params["PS_sig_width"]
#         S2S_outer_gap_width = params["PS_outer_gap_width"]
#         S2S_outer_gnd_width = params["PS_outer_gnd_width"]
#
#     ### taper type == pads to indivudual S2S design
#     elif taper_type == 2: #leave s2s params unchanged
#         pass
#
#     elif taper_type == 3: #custom widths for wscl
#         params["S2S_center_gnd_width"] = 200
#         params["S2S_inner_gap_width"] = 13
#         params["S2S_sig_width"] = 135
#         params["S2S_outer_gap_width"] = 20
#         params["S2S_outer_gnd_width"] = 39
#         S2S_center_gnd_width = params["S2S_center_gnd_width"]
#         S2S_inner_gap_width = params["S2S_inner_gap_width"]
#         S2S_sig_width = params["S2S_sig_width"]
#
#     #termination pad parameters
#     if termination == 35:
#         pad_t_center_gnd_width = 120
#         pad_t_inner_gap_width = 7.5
#         pad_t_sig_width = 100
#         pad_t_outer_gap_width = 7.5
#         pad_t_outer_gnd_width = 200
#         htr_width_local = params["htr_width_35"]
#
#     elif termination == 50:
#         pad_t_center_gnd_width = 100
#         pad_t_inner_gap_width = 20
#         pad_t_sig_width = 80
#         pad_t_outer_gap_width = 20
#         pad_t_outer_gnd_width = 86
#         htr_width_local = params["htr_width_50"]
#
#     elif termination == 65:
#         pad_t_center_gnd_width = 60
#         pad_t_inner_gap_width = 45
#         pad_t_sig_width = 70
#         pad_t_outer_gap_width = 45
#         pad_t_outer_gnd_width = 70
#         htr_width_local = params["htr_width_65"]
#
#     elif termination == 84:
#         pad_t_center_gnd_width = 80
#         pad_t_inner_gap_width = 20
#         pad_t_sig_width = 80
#         pad_t_outer_gap_width = 20
#         pad_t_outer_gnd_width = 80
#         htr_width_local = params["htr_width_84"]
#         center_slot_line_offset_MT3_term = center_slot_line_offset_MT3 - 0.5
#
#     elif termination == 0: #no termination specified (symmetric,_) case
#         pad_t_center_gnd_width = params["pad_center_gnd_width"]
#         pad_t_inner_gap_width = params["pad_inner_gap_width"]
#         pad_t_sig_width = params["pad_sig_width"]
#         pad_t_outer_gap_width = params["pad_outer_gap_width"]
#         pad_t_outer_gnd_width = params["pad_outer_gnd_width"]
#         pad_t_length = params["pad_length"]
#         trans_length_to_term = trans_length
#         center_slot_line_offset_MT3_term = center_slot_line_offset_MT3
#
#     else:
#         raise ValueError("Invalid termination resistance")
#
#     if termination != 0:
#         htr_length = params["htr_length"]
#         htr_connect_length = params["htr_connect_length"]
#         htr_connect_width = params["htr_connect_width"]
#         sc_length = params["sc_length"]
#         pad_t_length = params["pad_t_length"]
#         trans_length_to_term = params["trans_length_to_term"]
#
#         htr_width = params["htr_width"]
#         htr_further_params = {
#         "pad_center_gnd_width": params["htr_further_center_gnd_width"],
#         "pad_inner_gap_width": params["htr_further_inner_gap_width"],
#         "pad_sig_width": params["htr_further_sig_width"],
#         "pad_outer_gap_width": params["htr_further_outer_gap_width"],
#         "pad_outer_gnd_width": params["htr_further_outer_gnd_width"]
#         }
#
#         htr_closer_params = {
#             "pad_center_gnd_width": params["htr_closer_center_gnd_width"],
#             "pad_inner_gap_width": params["htr_closer_inner_gap_width"],
#             "pad_sig_width": params["htr_closer_sig_width"],
#             "pad_outer_gap_width": params["htr_closer_outer_gap_width"],
#             "pad_outer_gnd_width": params["htr_closer_outer_gnd_width"],
#         }
#
#         htr_connect_params = {
#             "pad_center_gnd_width": 0,
#             "pad_inner_gap_width": htr_further_params["pad_inner_gap_width"],
#             "pad_sig_width": params["htr_width"],
#             "pad_outer_gap_width": htr_closer_params["pad_inner_gap_width"],
#             "pad_outer_gnd_width": params["htr_width"],
#         }
#
#
#     if "pad_t_center_gnd_width" in params:  # param library termination override - for example for bonding
#         pad_t_center_gnd_width = params["pad_t_center_gnd_width"]
#         pad_t_inner_gap_width = params["pad_t_inner_gap_width"]
#         pad_t_sig_width = params["pad_t_sig_width"]
#         pad_t_outer_gap_width = params["pad_t_outer_gap_width"]
#         pad_t_outer_gnd_width = params["pad_t_outer_gnd_width"]
#         pad_t_length = params["pad_t_length"]
#         trans_length_to_term = params["trans_length"]
#
#
#
#     # param presets(legacy)
#     pad_params = {
#         "pad_center_gnd_width": params["pad_center_gnd_width"],
#         "pad_inner_gap_width": params["pad_inner_gap_width"],
#         "pad_sig_width": params["pad_sig_width"],
#         "pad_outer_gap_width": params["pad_outer_gap_width"],
#         "pad_outer_gnd_width": params["pad_outer_gnd_width"],
#         "pad_length": pad_length,
#     }
#     pad_PAD_params = {
#         "pad_center_gnd_width": pad_params["pad_center_gnd_width"] - 10,
#         "pad_inner_gap_width": pad_params["pad_inner_gap_width"] + 10,
#         "pad_sig_width": pad_params["pad_sig_width"] - 10,
#         "pad_outer_gap_width": pad_params["pad_outer_gap_width"] + 10,
#         "pad_outer_gnd_width": pad_params["pad_outer_gnd_width"] - 10,
#         "pad_length": pad_length - 10,
#     }
#     s2s_params = {
#         "pad_center_gnd_width": params["S2S_center_gnd_width"],
#         "pad_inner_gap_width": params["S2S_inner_gap_width"],
#         "pad_sig_width": params["S2S_sig_width"],
#         "pad_outer_gap_width": params["S2S_outer_gap_width"],
#         "pad_outer_gnd_width": params["S2S_outer_gnd_width"],
#         "pad_length": params["S2S_length"],
#     }
#     ps_params = {
#         "pad_center_gnd_width": params["PS_center_gnd_width"],
#         "pad_inner_gap_width": params["PS_inner_gap_width"],
#         "pad_sig_width": params["PS_sig_width"],
#         "pad_outer_gap_width": params["PS_outer_gap_width"],
#         "pad_outer_gnd_width": params["PS_outer_gnd_width"],
#         "pad_length": params["PS_length"],
#     }
#
#     components_along_path = []
#     sections_electrical_extend = []
#     s = sections_electrical_extend.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section (dummmy) for multiple components
#
#
#     # central column of vias
#     num_rows_contact = min(3, int(params["PS_MT1_center_gnd_width"] / (via_size_contact + gap_via_contact)) - 1)
#     offset_contact = -num_rows_contact * (via_size_contact + gap_via_contact) / 2 + gap_via_contact # = int(num_rows_contact/2 -1)*gap_via_contact + (num_rows_contact/2) *via_size_contact[1] #(num_rows_contact*via_size_contact[1] + (num_rows_contact-1)*gap_via_contact)/2# int(num_rows_contact / 2 - 1) * gap_via_contact + (num_rows_contact / 2) * via_size_contact[1]
#     if num_rows_contact == 1:  # hack fix
#         offset_contact = -via_size_contact / 2
#
#     for i in range(num_rows_contact):
#         components_along_path.append(ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT),
#                                                                    spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#
#                                                                    offset=(offset_contact + i * (via_size_contact + gap_via_contact))))
#
#     offset_via_1 = -num_rows_V1 * (via_size_top + gap_via_top) / 2 + gap_via_top
#     for i in range(num_rows_V1):
#         components_along_path.append(
#             ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                offset=(offset_via_1 + i * (via_size_top + gap_via_top))))
#
#     x1 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path)
#     p1 = gf.path.straight(length=params["PS_length"])
#     PS = gf.path.extrude(p1, x1)
#     _ = c << PS
#     _.movex(params["pad_length"] + params["trans_length"] + params["S2S_length"])
#
#
#
#     '''termination structures'''
#     if termination != 0:
#         x_offset = trans_length + pad_length + 2*params["S2S_length"] + PS_length + trans_length_to_term
#         htr_center = x_offset + pad_t_length/2
#
#         # heaters, _ = GSG_piece(params, htr_connect_width+2*htr_length, 0, 0, 0, 0, htr_width_local, layer_HTR)
#         # _ = c << heaters
#         # _.movex(htr_center - htr_width_local/2)
#         #
#         # htr_connect, _ = GSG_piece(params,htr_connect_width, htr_length, htr_connect_width, 0, 0, htr_connect_length, layer_HTR)
#         # _ = c << htr_connect
#         # _.movex(htr_center - htr_connect_length/2)
#
#         b = gf.Component("resistor")
#         heater_further, _ = GSG_piece(params, **htr_further_params, pad_length=htr_width, layer_MTX=HTR)
#         _ = b << heater_further
#         if not params["htr_flipped"]:  _.movex(htr_center + htr_length - 1.5*htr_width)
#         else:   _.movex(htr_center - (htr_length - 0.5*htr_width))
#
#         heater_closer, _ = GSG_piece(params, **htr_closer_params, pad_length=htr_width, layer_MTX=HTR)
#         _ = b << heater_closer
#         _.movex(htr_center - htr_width/2)
#
#         heater_connector, _ = SGS_piece(**htr_connect_params, pad_length=htr_length, layer_MT2=HTR)
#         _ = b << heater_connector
#         if not params["htr_flipped"]: _.movex(htr_center- htr_width/ 2)
#         else: _.movex(htr_center - (htr_length- 0.5*htr_width))
#
#         b = gf.geometry.fillet(b, radius=params["htr_fillet_radius"])
#         c.add_polygon(b)
#
#
#         #mt2 pads
#         pad_t_m2_out, x50 = GSG_piece(params, pad_t_center_gnd_width-10, pad_t_inner_gap_width+10, pad_t_sig_width-10, pad_t_outer_gap_width+10, pad_t_outer_gnd_width-10 , pad_t_length-10, MT2,
#                                       slot=False, layer_MTX_SLOT=MT2_SLOT, num_rows_slot = 3)
#         pad_out_ref = c << pad_t_m2_out
#         pad_out_ref.movex(trans_length + pad_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term + 5)
#
#         # pad_cu_out, x51 = GSG_piece(params, pad_t_center_gnd_width, pad_t_inner_gap_width, pad_t_sig_width, pad_t_outer_gap_width, pad_t_outer_gnd_width, pad_t_length, MT1)
#         # pad_out_ref = c << pad_cu_out
#         # pad_out_ref.movex(trans_length + pad_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term)
#
#
#         #termination vias
#         components_along_path_electrical = []
#         sections_electrical = []
#         _ = sections_electrical.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section for whatever reason (dummy section)
#
#         num_rows_via_contact_term = int(htr_connect_width/(via_size_contact+gap_via_contact)) -2
#         offset_via_2 = num_rows_via_contact_term * (via_size_contact + gap_via_contact)/2 - gap_via_contact
#         offset_sig = htr_closer_params["pad_center_gnd_width"]/2 + htr_closer_params["pad_inner_gap_width"] + htr_closer_params["pad_sig_width"] - 8
#         for i in range(num_rows_via_contact_term):
#             components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT), spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#                                                                               offset=(offset_sig + i * (via_size_contact + gap_via_contact))))
#             components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT), spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#                                                                               offset=-(offset_sig + i * (via_size_contact + gap_via_contact))))
#             components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT), spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#                                                                               offset= (-offset_via_2 + i * (via_size_contact + gap_via_contact)) ))
#
#         num_rows_via1_term = int(htr_connect_width / (via_size_1 + gap_via_1)) -2
#         offset_via_1 = num_rows_via1_term * (via_size_1 + gap_via_1) / 2 - gap_via_1
#         for i in range(num_rows_via1_term):
#             components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_1, via_size_1), layer=VIA1), spacing=via_size_1 + gap_via_1, padding=via_size_1 + gap_via_1,
#                                                                               offset=(offset_sig + i * (via_size_1 + gap_via_1))))
#             components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_1, via_size_1), layer=VIA1), spacing=via_size_1 + gap_via_1, padding=via_size_1 + gap_via_1,
#                                                                               offset=-(offset_sig + i * (via_size_1 + gap_via_1))))
#             components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_1, via_size_1), layer=VIA1), spacing=via_size_1 + gap_via_1, padding=via_size_top + gap_via_1,
#                                                                        offset=(-offset_via_1 + i * (via_size_1 + gap_via_1)) ))
#
#         b = gf.Component("M1_term_pads")
#         #m1 pads
#         offset_m1 = offset_sig+htr_connect_width/2 - (2*via_size_1 + 2*gap_via_1)
#         m1_pad =  gf.components.rectangle(size=[htr_connect_length, htr_connect_width], layer=MT1)
#         _ = b << m1_pad
#         _.move([htr_center- htr_connect_length/2, -htr_connect_width/2+offset_m1])
#         _ = b << m1_pad
#         _.move([htr_center - htr_connect_length / 2, -htr_connect_width/2-offset_m1])
#         _ = b << m1_pad
#         _.move([htr_center - htr_connect_length / 2, -htr_connect_width/2])
#
#         b = gf.geometry.fillet(b, radius=params["htr_connect_fillet_radius"])
#         c.add_polygon(b)
#         c.add_polygon(b, layer=HTR)
#
#         x1_2 = gf.CrossSection(sections=sections_electrical, components_along_path=components_along_path_electrical)
#         p1_2 = gf.path.straight(length=htr_connect_length)
#         PS_e = gf.path.extrude(p1_2, x1_2)
#         _ = c << PS_e
#         _.movex(htr_center - htr_connect_length/2)#params["pad_length"] + 2*params["S2S_length"] + params["trans_length"] + params["PS_length"] + trans_length_to_term + 15)
#
#
#
#         #metal2 pad slots
#         if cheese:
#             c_M2_cheese = gf.components.rectangle(size=cheese_slot_size_MT1, layer=MT2_SLOT)
#             c_M2_cheese_pair = gf.Component("c_M2_cheese_pair")
#             _ = c_M2_cheese_pair << c_M2_cheese
#             _.move((htr_center, pad_t_center_gnd_width / 4 - 7))
#             _ = c_M2_cheese_pair << c_M2_cheese
#             _.move((htr_center, pad_t_center_gnd_width / 4 - 7)).mirror_y()
#
#             _ = c << c_M2_cheese_pair
#             _ = c << c_M2_cheese_pair
#             _.movey(pad_t_center_gnd_width + 16)
#             _ = c << c_M2_cheese_pair
#             _.movey(pad_t_center_gnd_width + 16).mirror_y()
#
#
#     if params["gnds_shorted"]:
#        g_extend, _ = GSG_piece(params, 0, pad_t_center_gnd_width/2 + pad_t_inner_gap_width,  pad_t_sig_width, pad_t_outer_gap_width, pad_t_outer_gnd_width, sc_length, layer_MTX)
#        _ = c << g_extend
#        _.movex(x_offset + pad_t_length)
#
#        g_connect, _ = GSG_piece(params, pad_t_center_gnd_width, 0, pad_t_inner_gap_width + pad_t_sig_width , 0, 0, sc_length, layer_MTX)
#        _ = c << g_connect
#        _.movex(x_offset + pad_t_length + sc_length)
#
#
#     # Input contact pads
#     pad_in, x1 = GSG_piece(params, **pad_params, layer_MTX=layer_MTX)
#     c << pad_in
#
#     pad_in_PAD, _ = GSG_piece(params, **pad_PAD_params, layer_MTX=PAD)
#     pad_in_PAD_ref = c << pad_in_PAD
#     pad_in_PAD_ref.movex(5)
#
#     # S2S 1
#     s2s_1, x2 = GSG_piece(params, **s2s_params, layer_MTX=layer_MTX, layer_MTX_SLOT = MT3_SLOT, slot=cheese, num_rows_slot=3, )
#     s2s_1_ref = c << s2s_1
#     s2s_1_ref.movex(trans_length + pad_length)
#
#
#     # PS region
#     #top metal layer
#     ps, _ = GSG_piece(params, **ps_params, layer_MTX=layer_MTX, layer_MTX_SLOT = MT3_SLOT, slot=cheese, num_rows_slot=2)
#     ps_ref = c << ps
#     ps_ref.movex(trans_length + pad_length + s2s_params["pad_length"])
#
#
#     # ps, _ = GSG_piece(params, pad_center_gnd_width=PS_MT1_center_gnd_width,
#     #                   pad_inner_gap_width=PS_MT1_inner_gap_width,
#     #                   pad_sig_width=PS_MT1_sig_width,
#     #                   pad_outer_gap_width=0,
#     #                   pad_outer_gnd_width=0,
#     #                   pad_length=params["PS_length"] + 2*params["extension_electrical"], layer_MTX=MT2, slot=True, layer_MTX_SLOT=MT2_SLOT, num_rows_slot = 2)
#     # ps_ref = c << ps
#     # ps_ref.movex(trans_length + pad_length + s2s_params["pad_length"] - params["extension_electrical"])
#
#     #M1 layer
#     ps, _ = GSG_piece(params, pad_center_gnd_width=PS_MT1_center_gnd_width,
#                       pad_inner_gap_width=PS_MT1_inner_gap_width,
#                       pad_sig_width=PS_MT1_sig_width,
#                       pad_outer_gap_width=0,
#                       pad_outer_gnd_width=0,
#                       pad_length=params["PS_length"], layer_MTX=MT1, slot=cheese, layer_MTX_SLOT=MT1_SLOT, num_rows_slot = 2)
#     ps_ref = c << ps
#     ps_ref.movex(trans_length + pad_length + s2s_params["pad_length"])
#
#
#
#     '''*******************special extension at top for M3 to M2 transition******************************'''
#     #
#     # M1_extend = gf.Component("M1_extend")
#     # # bottom left, clockwise
#     # x_M1_extend = [-3,
#     #                -3,
#     #                0,
#     #                0]
#     # y_M1_extend = [-params["PS_MT1_center_gnd_width"] / 2,
#     #                params["PS_MT1_center_gnd_width"] / 2,
#     #                params["PS_MT1_center_gnd_width"] / 2,
#     #                -params["PS_MT1_center_gnd_width"] / 2]
#     # #M1_extend.add_polygon([x_M1_extend, y_M1_extend], layer=MT2)
#     #
#     # components_list = []
#     # _ = None
#     #
#     #
#     # if params["PS_MT1_center_gnd_width"] >= 2.5:
#     #     V1 =  gf.components.rectangle(size=[params["via_size_top"], params["via_size_top"]], layer=VIA1)
#     #     for i in range(4):
#     #         components_list.append(V1)
#     #     M1_extend_grid = gf.grid(
#     #         components_list,
#     #         spacing=(params["gap_via_top"], params["gap_via_top"]),
#     #         separation=True,
#     #         shape=(2, 2),
#     #         align_x="x",
#     #         align_y="y",
#     #         edge_x="x",
#     #         edge_y="ymax",
#     #     )
#     #
#     #     _ = M1_extend << M1_extend_grid
#     #     _.move((-3 + params["gap_via_top"] + params["via_size_top"] / 2, - (params["gap_via_top"] / 2 + params["via_size_top"])))
#     #
#     # elif params["PS_MT1_center_gnd_width"] < 2.5:
#     #     components_list = []
#     #     V1 = gf.components.rectangle(size=[params["via_size_top"], params["via_size_top"]], layer=VIA1)
#     #     for i in range(2):
#     #         components_list.append(V1)
#     #     M1_extend_grid = gf.grid(
#     #         components_list,
#     #         spacing=(params["gap_via_top"], params["gap_via_top"]),
#     #         separation=True,
#     #         shape=(1, 2),
#     #         align_x="x",
#     #         align_y="y",
#     #         edge_x="x",
#     #         edge_y="ymax",
#     #     )
#     #     _ = M1_extend << M1_extend_grid
#     #     _.move((-3 + params["gap_via_top"] + params["via_size_top"] / 2, -params["via_size_top"]/2))
#     #
#     # _ = c << M1_extend
#     # _.movex(params["pad_length"] + params["trans_length"] + params["S2S_length"])
#     #
#     # _ = c << M1_extend
#     # _.rotate(180)
#     # _.movex(params["pad_length"] + params["trans_length"] + params["S2S_length"] + params["PS_length"])
#
#     '''***************************************************************************************'''
#
#     # Transition 1
#     taper1 = gf.components.taper_cross_section_linear(x1, x2, length=trans_length)
#     taper1_ref = c << taper1
#     taper1_ref.movex(pad_length)
#
#     # Transition 1 cheese outside electrodes
#     if cheese:
#         pad_offset = (pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2)
#         s2s_offset = (S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width / 2)
#         bl = (pad_length-1*border_slot_MT3, pad_offset - pad_sig_width/2)
#         br = (pad_length + trans_length + 1*border_slot_MT3, s2s_offset - S2S_sig_width/2)
#         tr = (pad_length + trans_length + 1.2*border_slot_MT3, s2s_offset + S2S_sig_width/2 -0.5)
#         tl = (pad_length-1*border_slot_MT3, pad_offset + pad_sig_width/2)
#         poly = [bl, br, tr, tl]
#         c_cheese = PEO_cheese(poly, slot_size=cheese_slot_size_MT3, gap_slot=gap_slot_MT3, border_slot=border_slot_MT3, layer_cheese=MT3_SLOT, rotation_degrees=math.degrees(math.atan((s2s_offset - pad_offset) / trans_length)))
#         cheese_ref = c << c_cheese
#         cheese_ref.move((-cheese_shift_y, cheese_shift_x))
#         cheese_ref_bot = c << c_cheese
#         cheese_ref_bot.mirror_y().move((-cheese_shift_y, -cheese_shift_x))
#
#         # Transition 1 center
#         gnd_50_length = -(trans_length/(pad_center_gnd_width/2)) * (25 - pad_center_gnd_width/2)
#         gnd_30_length = -(trans_length/(pad_center_gnd_width/2)) * (15 - pad_center_gnd_width/2)
#
#         bl = (pad_length-1*border_slot_MT3, -pad_center_gnd_width / 2)
#         br = (pad_length + gnd_50_length + border_slot_MT3 + 1.5*gap_slot_MT3, -25)#-S2S_center_gnd_width / 2)
#         tr = (pad_length + gnd_50_length + border_slot_MT3 + 1.5*gap_slot_MT3, 25)#S2S_center_gnd_width / 2)
#         tl = (pad_length-1*border_slot_MT3, pad_center_gnd_width / 2)
#         poly = [bl, br, tr, tl]
#         _ = c << PEO_cheese(poly, slot_size=cheese_slot_size_MT3, gap_slot=gap_slot_MT3, border_slot=border_slot_MT3, layer_cheese=MT3_SLOT, rotation_degrees=0)
#         _.move((-cheese_shift_y, 0))
#
#         #single line in thin part
#         if pad_t_center_gnd_width > 30:
#             components_along_gnd_path = []
#             for i in range(1):
#                 components_along_gnd_path.append(
#                     ComponentAlongPath(component=gf.c.via(size=cheese_slot_size_MT3, layer=MT3_SLOT), spacing=cheese_slot_size_MT3[0] + gap_slot_MT3, padding=gap_slot_MT3, offset=0))  # *(4+10) - num_rows_trans_slot*(4+10)/2 ))
#             #p = gf.path.straight(length=1.5*(gnd_30_length-gnd_50_length)-center_slot_line_offset_MT3, npoints=2)
#             p = gf.path.straight(length=trans_length-gnd_50_length - center_slot_line_offset_MT3, npoints=2)
#             x1 = gf.CrossSection(components_along_path=components_along_gnd_path)
#             generate = gf.path.extrude(p, x1)
#             _ = c << generate
#             _.movex(pad_length + gnd_50_length - center_slot_line_offset_MT3 + 3)
#
#
#
#     # S2S 2
#     s2s_2, x4 = GSG_piece(params, **s2s_params, layer_MTX=layer_MTX, layer_MTX_SLOT = MT3_SLOT, slot=True, num_rows_slot=3)
#     s2s_2_ref = c << s2s_2
#     s2s_2_ref.movex(trans_length + pad_length + s2s_params["pad_length"] + ps_params["pad_length"])
#
#     # output contact pads
#     pad_t_out, x5 = GSG_piece(params, pad_t_center_gnd_width, pad_t_inner_gap_width, pad_t_sig_width, pad_t_outer_gap_width, pad_t_outer_gnd_width, pad_t_length, layer_MTX)
#     pad_out_ref = c << pad_t_out
#     pad_out_ref.movex(trans_length + pad_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term)
#
#     pad_out_PAD, _ = GSG_piece(params, pad_t_center_gnd_width - 10, pad_t_inner_gap_width + 10, pad_t_sig_width - 10, pad_t_outer_gap_width + 10, pad_t_outer_gnd_width - 10, pad_t_length - 10, layer_PAD)
#     pad_out_PAD_ref = c << pad_out_PAD
#     pad_out_PAD_ref.movex(5 + trans_length + pad_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term)
#
# # Transition 2
#     taper2 = gf.components.taper_cross_section_linear(x4, x5, length=trans_length_to_term)
#     taper2_ref = c << taper2
#     taper2_ref.movex(pad_length + trans_length + params["PS_length"] + 2*params["S2S_length"])
#
#     if cheese:
#         #Transition 2 edges
#         pad_offset = (pad_t_center_gnd_width / 2 + pad_t_inner_gap_width + pad_t_sig_width / 2)
#         s2s_offset = (S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width / 2)
#         bl = (pad_t_length-1*border_slot_MT3, pad_offset - pad_sig_width/2)
#         br = (pad_t_length + trans_length_to_term + 1*border_slot_MT3, s2s_offset - S2S_sig_width/2)
#         tr = (pad_t_length + trans_length_to_term + 1.2*border_slot_MT3, s2s_offset + S2S_sig_width/2-0.5)
#         tl = (pad_t_length-1*border_slot_MT3, pad_offset + pad_sig_width/2)
#         poly = [bl, br, tr, tl]
#         c_cheese = PEO_cheese(poly, slot_size=cheese_slot_size_MT3, gap_slot=gap_slot_MT3, border_slot=border_slot_MT3,  layer_cheese=MT3_SLOT, rotation_degrees=(math.degrees(math.atan((s2s_offset - pad_offset) / trans_length_to_term))))
#         cheese_ref = c << c_cheese
#         cheese_ref.rotate(180)
#         cheese_ref.movex(pad_length + trans_length + 2*params["S2S_length"] + params["PS_length"] + trans_length_to_term + pad_t_length)
#         #cheese_ref.move((cheese_shift_y, cheese_shift_x))
#         cheese_ref_bot = c << c_cheese
#         cheese_ref_bot.mirror_y().move((cheese_shift_y, -cheese_shift_x))
#         cheese_ref_bot.rotate(180)
#         cheese_ref_bot.movex(pad_length + trans_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term + pad_t_length)
#
#         #Transition 2 center
#         gnd_50_length = -(trans_length_to_term/(pad_t_center_gnd_width/2)) * (25 - pad_t_center_gnd_width/2)
#         gnd_30_length = -(trans_length_to_term/(pad_t_center_gnd_width/2)) * (15 - pad_t_center_gnd_width/2)
#
#         bl = (pad_t_length-1*border_slot_MT3, -pad_t_center_gnd_width / 2)
#         br = (pad_t_length + gnd_50_length + border_slot_MT3 + 1.5 * gap_slot_MT3, -25)#-S2S_center_gnd_width / 2)
#         tr = (pad_t_length + gnd_50_length + border_slot_MT3 + 1.5 * gap_slot_MT3, 25)#S2S_center_gnd_width / 2)
#         tl = (pad_t_length-1*border_slot_MT3, pad_t_center_gnd_width / 2)
#         poly = [bl, br, tr, tl]
#         _ = c << PEO_cheese(poly, slot_size=cheese_slot_size_MT3, gap_slot=gap_slot_MT3, border_slot=border_slot_MT3, layer_cheese=MT3_SLOT, rotation_degrees=0)
#         _.rotate(180)
#         _.movex(pad_length + trans_length + 2*params["S2S_length"] + params["PS_length"] + trans_length_to_term + pad_t_length)
#         _.move((cheese_shift_y, 0))
#
#         #single line in thin part
#         if pad_t_center_gnd_width > 30:
#             components_along_gnd_path = []
#             for i in range(1):
#                 components_along_gnd_path.append(
#                     ComponentAlongPath(component=gf.c.via(size=cheese_slot_size_MT3, layer=MT3_SLOT), spacing=cheese_slot_size_MT3[0] + gap_slot_MT3, padding=gap_slot_MT3/2, offset=0))  # *(4+10) - num_rows_trans_slot*(4+10)/2 ))
#             #p = gf.path.straight(length=1.4*(gnd_30_length-gnd_50_length) - center_slot_line_offset_MT3, npoints=2)
#             p = gf.path.straight(length=trans_length_to_term-gnd_50_length - center_slot_line_offset_MT3_term + center_slot_line_length_delta_MT3_term-20, npoints=2)
#             x1 = gf.CrossSection(components_along_path=components_along_gnd_path)
#             generate = gf.path.extrude(p, x1)
#             _ = c << generate
#             _.rotate(180)
#             _.movex(pad_length + trans_length + 2*params["S2S_length"] + params["PS_length"] + trans_length_to_term - gnd_50_length + center_slot_line_offset_MT3_term - 10.5)
#
#
#    #dummy fill block
#     MT_dummy_width = params["pad_center_gnd_width"] + 2*params["pad_inner_gap_width"] + 2*params["pad_sig_width"]
#     MT_dummy_length = params["pad_length"] + params["trans_length"] + 2*params["S2S_length"] + params["PS_length"] + trans_length_to_term + pad_t_length
#     for i in [MT1_DUMMY_BLOCK, MT2_DUMMY_BLOCK]:#, MT3_DUMMY_BLOCK]:
#         _ = c << gf.components.rectangle(size=(MT_dummy_length+2*MT_dummy_margin, MT_dummy_width+2*MT_dummy_margin), layer=i)
#         _.move((-MT_dummy_margin, -MT_dummy_margin - MT_dummy_width/2))
#
#     mid_x = pad_length + trans_length + s2s_params["pad_length"]
#     c.add_port(
#         name="e_up",
#         center=(mid_x, -params["PS_MT1_center_gnd_width"] / 2 - params["PS_MT1_inner_gap_width"] / 2), #used to be S2S
#         width=1,
#         orientation=0,
#         layer=layer_MTX,
#         port_type="electrical",
#     )
#     c.add_port(
#         name="e_low",
#         center=(mid_x, params["PS_MT1_center_gnd_width"] / 2 + params["PS_MT1_inner_gap_width"] / 2),
#         width=1,
#         orientation=0,
#         layer=layer_MTX,
#         port_type="electrical",
#     )
#
#     return c


# @gf.cell #unused for SilTerra
# def GSGSG_MT2(PS_length, trans_length, taper_type, sig_trace, params:dict, gnds_shorted = False, termination=0, ps_config="default"): #no heaters
#     """
#     New GSGSG for Crealights with PS taper
#     Parameter overrides are kept in this function for ease of use when calling from CSV params
#     The below are overrides and specifications for GSGSG_MT2
#     :param PS_length
#     :param trans_length
#     :param params
#
#     :param taper_type: s2s overrides, 1 or 2
#     :param sig_trace: ps overrides, "narrow" "medium" "wide"
#     :param ps_config: ps parameter overrides: "default"(default), "narrow_custom" "medium_custom" "wide_custom": - unused as of 9/16/2025
#     :param termination:
#     if termination = 0, the terminating electrode pads will match the default pads for a symmetric electrode structure
#             if termination = 35, 50, 65, output pad geometry will be changed to the specified resistance. Also, heaters between the termination pads will be added.
#                 also, these additional parameters must be specified in parameter dictionary: "htr_width_x", "htr_length", "htr_connect_length",
#                                                                                             "sc_length", "pad_t_length",  "trans_length_to_term"
#     :param gnds_shorted: bool, if True ground shorting structure will be added
#     :param gsgsg_variant: almost obsolete, only used to specify SGS_MT2_DC in DOE8
#     :param config: standard (default), batch, compact - used only in SGS_MT2_DC
#     :return:
#     """
#
#     via_place_version = 2 #1 or 2 #v1 deprecated
#
#     #param overrides
#     if "TWEdes" not in params:
#         TWEdes = 0
#     else:
#         TWEdes = params["TWEdes"]
#
#     if TWEdes == "PCH_v4_70":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65  # 200-7.5,
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 40
#         params["S2S_outer_gap_width"] = 5
#         params["S2S_outer_gnd_width"] = 60
#         #S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 40
#         params["PS_outer_gap_width"] = 15
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_70-":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65  # 200-7.5,
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 40
#         params["S2S_outer_gap_width"] = 5
#         params["S2S_outer_gnd_width"] = 60
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 50
#         params["PS_outer_gap_width"] = 15
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_70+":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 40
#         params["S2S_outer_gap_width"] = 5
#         params["S2S_outer_gnd_width"] = 60
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 30
#         params["PS_outer_gap_width"] = 15
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_84":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 20
#         params["S2S_outer_gap_width"] = 4.5
#         params["S2S_outer_gnd_width"] = 62.5
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 20
#         params["PS_outer_gap_width"] = 17
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_84-":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 20
#         params["S2S_outer_gap_width"] = 4.5
#         params["S2S_outer_gnd_width"] = 62.5
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 25
#         params["PS_outer_gap_width"] = 17
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_84+":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 20
#         params["S2S_outer_gap_width"] = 4.5
#         params["S2S_outer_gnd_width"] = 62.5
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 15
#         params["PS_outer_gap_width"] = 17
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_95":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 25
#         params["S2S_outer_gap_width"] = 11
#         params["S2S_outer_gnd_width"] = 67
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 15
#         params["PS_outer_gap_width"] = 28
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_95-":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 25
#         params["S2S_outer_gap_width"] = 11
#         params["S2S_outer_gnd_width"] = 67
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 20
#         params["PS_outer_gap_width"] = 28
#         params["PS_outer_gnd_width"] = 50
#
#     if TWEdes == "PCH_v4_95+":
#         params["pad_center_gnd_width"] = 65
#         params["pad_inner_gap_width"] = 35
#         params["pad_sig_width"] = 65
#         params["pad_outer_gap_width"] = 35
#         params["pad_outer_gnd_width"] = 65
#         params["pad_length"] = 60
#
#         params["S2S_center_gnd_width"] = 50
#         params["S2S_inner_gap_width"] = 16
#         params["S2S_sig_width"] = 25
#         params["S2S_outer_gap_width"] = 11
#         params["S2S_outer_gnd_width"] = 67
#         # S2S_length = 50
#
#         params["PS_center_gnd_width"] = 50
#         params["PS_inner_gap_width"] = 16
#         params["PS_sig_width"] = 10
#         params["PS_outer_gap_width"] = 28
#         params["PS_outer_gnd_width"] = 50
#
#
#
#     ### PS signal traces == narrow, medium or wide
#     PS_center_gnd_width = params["PS_center_gnd_width"]  # we should just call the params directly when needed..
#     PS_inner_gap_width = params["PS_inner_gap_width"]
#     PS_sig_width = params["PS_sig_width"]
#     PS_outer_gap_width = params["PS_outer_gap_width"]
#     PS_outer_gnd_width = params["PS_outer_gnd_width"]
#
#     # PS_MT1_center_gnd_width = params["PS_MT1_center_gnd_width"]
#     # PS_MT1_inner_gap_width = params["PS_MT1_inner_gap_width"]
#     # PS_MT1_sig_width = params["PS_MT1_sig_width"]
#     # PS_MT1_outer_gap_width = params["PS_MT1_outer_gap_width"]
#     # PS_MT1_outer_gnd_width = params["PS_MT1_outer_gnd_width"]
#     PS_trans_length = float(params["PS_trans_length"]) #somehow got cast to numpy.int...
#
#     # pad_MT1_center_gnd_width = params["pad_MT1_center_gnd_width"]
#     # pad_MT1_inner_gap_width = params["pad_MT1_inner_gap_width"]
#     # pad_MT1_sig_width = params["pad_MT1_sig_width"]
#     # pad_MT1_outer_gap_width = params["pad_MT1_outer_gap_width"]
#     # pad_MT1_outer_gnd_width = params["pad_MT1_outer_gnd_width"]
#
#     S2S_center_gnd_width = params["S2S_center_gnd_width"]
#     S2S_inner_gap_width = params["S2S_inner_gap_width"]
#     S2S_sig_width = params["S2S_sig_width"]
#     S2S_outer_gap_width = params["S2S_outer_gap_width"]
#     S2S_outer_gnd_width = params["S2S_outer_gnd_width"]
#     S2S_length = params["S2S_length"]
#
#     pad_center_gnd_width = params["pad_center_gnd_width"]
#     pad_inner_gap_width = params["pad_inner_gap_width"]
#     pad_sig_width = params["pad_sig_width"]
#     pad_outer_gap_width = params["pad_outer_gap_width"]
#     pad_outer_gnd_width = params["pad_outer_gnd_width"]
#     pad_length = params["pad_length"]
#
#     sc_length = params["sc_length"]
#     layer_MT2 = params["layer_MT2"]
#     spacing_x = params["spacing_x"]
#     spacing_y = params["spacing_y"]
#     DC_pad_size_x = params["DC_pad_size_x"]
#     DC_pad_size_y = params["DC_pad_size_y"]
#     pads_rec_gap = params["pads_rec_gap"]
#     layer_MT2 = params["layer_MT2"]
#     layer_PAD = params["layer_PAD"]
#     layer_HTR = params["layer_HTR"]
#     layer_VIA2 = params["layer_VIA2"]
#
#     via_size_top = params["via_size_top"]
#     gap_via_top = params["gap_via_top"]
#     via_size_1 = params["via_size_1"]
#     gap_via_1 = params["gap_via_1"]
#     via_size_contact = params["via_size_contact"][0]
#     gap_via_contact = params["gap_via_contact"]
#     num_rows_V1 = params["num_rows_V1"]
#
#
#     #Override PS parameters for custom configurations - originally used for S21?
#     #unused as of 9/15/2025
#     if ps_config == "narrow_custom":
#         PS_center_gnd_width = 200
#         PS_inner_gap_width = 13
#         PS_sig_width = 10
#         PS_outer_gap_width = 140
#         PS_outer_gnd_width = 50
#     elif ps_config == "medium_custom":
#         PS_center_gnd_width = 200
#         PS_inner_gap_width = 13
#         PS_sig_width = 60
#         PS_outer_gap_width = 90
#         PS_outer_gnd_width = 50
#     elif ps_config == "wide_custom":
#         PS_center_gnd_width = 200
#         PS_inner_gap_width = 13
#         PS_sig_width = 140
#         PS_outer_gap_width = 10
#         PS_outer_gnd_width = 140
#     elif ps_config == "default":
#         pass
#     else:
#         raise ValueError("Invalid ps_config")
#
#     ### S2S parameters "taper type"
#     if taper_type == 1:
#         S2S_center_gnd_width_local = PS_center_gnd_width
#         S2S_inner_gap_width_local = PS_inner_gap_width
#         S2S_sig_width_local = PS_sig_width
#         S2S_outer_gap_width_local = PS_outer_gap_width
#         S2S_outer_gnd_width_local = PS_outer_gnd_width
#     ### taper type == pads to individual S2S design
#     elif taper_type == 2:
#         S2S_center_gnd_width_local = S2S_center_gnd_width
#         S2S_inner_gap_width_local = S2S_inner_gap_width
#         S2S_sig_width_local = S2S_sig_width
#         S2S_outer_gap_width_local = S2S_outer_gap_width
#         S2S_outer_gnd_width_local = S2S_outer_gnd_width
#     else:
#         raise ValueError("check taper_type")
#
#     #termination pad parameters
#     if termination == 70:
#         pad_t_center_gnd_width = 50
#         pad_t_inner_gap_width = 18
#         pad_t_sig_width = 95
#         pad_t_outer_gap_width = 18
#         pad_t_outer_gnd_width = 65
#         T_width=26
#         params["htr_further_center_gnd_width"] = 0
#         params["htr_further_inner_gap_width"] = 22
#         params["htr_further_sig_width"] = 2 * T_width + 19
#         params["htr_further_outer_gap_width"] = 0
#         params["htr_further_outer_gnd_width"] = 0
#
#         params["htr_closer_center_gnd_width"] = 96
#         params["htr_closer_inner_gap_width"] = 19
#         params["htr_closer_sig_width"] = 43 + 10
#         params["htr_closer_outer_gap_width"] = 0
#         params["htr_closer_outer_gnd_width"] = 0
#
#         params["htr_length"] = 80+T_width
#         params["htr_width"] = T_width
#         params["htr_fillet_radius"] = 2
#         params["htr_flipped"] = True
#
#         params["htr_connect_width"] = 9
#         params["htr_connect_length"] = 30
#
#     elif termination == 84:
#         pad_t_center_gnd_width = 50
#         pad_t_inner_gap_width = 28
#         pad_t_sig_width = 80
#         pad_t_outer_gap_width = 28
#         pad_t_outer_gnd_width = 65
#         T_width = 24
#         params["htr_further_center_gnd_width"] = 0
#         params["htr_further_inner_gap_width"] = 23
#         params["htr_further_sig_width"] = 2 * T_width + 21
#         params["htr_further_outer_gap_width"] = 0
#         params["htr_further_outer_gnd_width"] = 0
#
#         params["htr_closer_center_gnd_width"] = 94
#         params["htr_closer_inner_gap_width"] = 21
#         params["htr_closer_sig_width"] = 42 + 10
#         params["htr_closer_outer_gap_width"] = 0
#         params["htr_closer_outer_gnd_width"] = 0
#
#         params["htr_length"] = 90 + T_width
#         params["htr_width"] = T_width
#         params["htr_fillet_radius"] = 2
#         params["htr_flipped"] = True
#
#         params["htr_connect_width"] = 9
#         params["htr_connect_length"] = 28
#
#     elif termination == 95:
#         pad_t_center_gnd_width = 65
#         pad_t_inner_gap_width = 35
#         pad_t_sig_width = 65
#         pad_t_outer_gap_width = 35
#         pad_t_outer_gnd_width = 65
#         htr_width_local = 15
#
#         params["htr_connect_width"] = 9
#
#     elif termination == 0: #no termination specified (symmetric,_) case
#         pad_t_center_gnd_width = params["pad_center_gnd_width"]
#         pad_t_inner_gap_width = params["pad_inner_gap_width"]
#         pad_t_sig_width = params["pad_sig_width"]
#         pad_t_outer_gap_width = params["pad_outer_gap_width"]
#         pad_t_outer_gnd_width = params["pad_outer_gnd_width"]
#         pad_t_length = params["pad_length"]
#         trans_length_to_term = trans_length
#
#     else:
#         raise ValueError("Invalid termination resistance")
#
#     if termination !=0:
#         htr_length = params["htr_length"]
#         htr_connect_length = params["htr_connect_length"]
#         htr_connect_width = params["htr_connect_width"]
#         sc_length = params["sc_length"]
#         pad_t_length = params["pad_t_length"]
#         trans_length_to_term = params["trans_length_to_term"]
#
#         htr_width = params["htr_width"]
#         htr_further_params = {
#         "pad_center_gnd_width": params["htr_further_center_gnd_width"],
#         "pad_inner_gap_width": params["htr_further_inner_gap_width"],
#         "pad_sig_width": params["htr_further_sig_width"],
#         "pad_outer_gap_width": params["htr_further_outer_gap_width"],
#         "pad_outer_gnd_width": params["htr_further_outer_gnd_width"]
#         }
#
#         htr_closer_params = {
#             "pad_center_gnd_width": params["htr_closer_center_gnd_width"],
#             "pad_inner_gap_width": params["htr_closer_inner_gap_width"],
#             "pad_sig_width": params["htr_closer_sig_width"],
#             "pad_outer_gap_width": params["htr_closer_outer_gap_width"],
#             "pad_outer_gnd_width": params["htr_closer_outer_gnd_width"],
#         }
#
#         htr_connect_params = {
#             "pad_center_gnd_width": 0,
#             "pad_inner_gap_width": htr_further_params["pad_inner_gap_width"],
#             "pad_sig_width": params["htr_width"],
#             "pad_outer_gap_width": htr_closer_params["pad_inner_gap_width"],
#             "pad_outer_gnd_width": params["htr_width"],
#         }
#
#
#     if "pad_t_center_gnd_width" in params: #param library termination override - for example for bonding
#         pad_t_center_gnd_width = params["pad_t_center_gnd_width"]
#         pad_t_inner_gap_width = params["pad_t_inner_gap_width"]
#         pad_t_sig_width = params["pad_t_sig_width"]
#         pad_t_outer_gap_width = params["pad_t_outer_gap_width"]
#         pad_t_outer_gnd_width = params["pad_t_outer_gnd_width"]
#         pad_t_length = params["pad_t_length"]
#         trans_length_to_term = trans_length
#
#     c = gf.Component()
#
#     # input gnd pads short circuit
#     if gnds_shorted:
#         sc_extend, x0 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width + pad_sig_width, 0, pad_outer_gap_width, pad_outer_gnd_width, sc_length, layer_MT2)
#         _ = c << sc_extend
#         _.movex(-sc_length)
#         sc_bridge, x0 = SGS_piece(pad_center_gnd_width, 0, pad_inner_gap_width + pad_sig_width + pad_outer_gap_width, 0, pad_outer_gnd_width, sc_length, layer_MT2)
#         _ = c << sc_bridge
#         _.movex(-sc_length * 2)
#
#     #heaters, only if special termination specified
#     if termination != 0:
#         x_offset = trans_length + pad_length + 2 * params["S2S_length"] + PS_length + trans_length_to_term
#         htr_center = x_offset + pad_t_length / 2
#
#         # heaters, _ = GSG_piece(params, htr_connect_width+2*htr_length, 0, 0, 0, 0, htr_width_local, layer_HTR)
#         # _ = c << heaters
#         # _.movex(htr_center - htr_width_local/2)
#         #
#         # htr_connect, _ = GSG_piece(params,htr_connect_width, htr_length, htr_connect_width, 0, 0, htr_connect_length, layer_HTR)
#         # _ = c << htr_connect
#         # _.movex(htr_center - htr_connect_length/2)
#         c_heater = gf.Component("heater")
#         b = gf.Component("resistor")
#         heater_further, _ = GSG_piece(params, **htr_further_params, pad_length=htr_width, layer_MTX=HTR)
#         _ = b << heater_further
#         if not params["htr_flipped"]:
#             _.movex(htr_center + htr_length - 1.5 * htr_width)
#         else:
#             _.movex(htr_center - (htr_length - 0.5 * htr_width))
#
#         heater_closer, _ = GSG_piece(params, **htr_closer_params, pad_length=htr_width, layer_MTX=HTR)
#         _ = b << heater_closer
#         _.movex(htr_center - htr_width / 2)
#
#         heater_connector, _ = SGS_piece(**htr_connect_params, pad_length=htr_length, layer_MT2=HTR)
#         _ = b << heater_connector
#         if not params["htr_flipped"]:
#             _.movex(htr_center - htr_width / 2)
#         else:
#             _.movex(htr_center - (htr_length - 0.5 * htr_width))
#
#         b = gf.geometry.fillet(b, radius=params["htr_fillet_radius"])
#         c_heater.add_polygon(b)
#
#         # mt2 pads
#         # pad_t_m2_out, x50 = GSG_piece(params, pad_t_center_gnd_width - 10, pad_t_inner_gap_width + 10, pad_t_sig_width - 10, pad_t_outer_gap_width + 10, pad_t_outer_gnd_width - 10, pad_t_length - 10, MT2,
#         #                               slot=False, layer_MTX_SLOT=MT2_SLOT, num_rows_slot=3)
#         # pad_out_ref = c << pad_t_m2_out
#         # pad_out_ref.movex(trans_length + pad_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term + 5)
#
#         # pad_cu_out, x51 = GSG_piece(params, pad_t_center_gnd_width, pad_t_inner_gap_width, pad_t_sig_width, pad_t_outer_gap_width, pad_t_outer_gnd_width, pad_t_length, MT1)
#         # pad_out_ref = c << pad_cu_out
#         # pad_out_ref.movex(trans_length + pad_length + 2 * params["S2S_length"] + params["PS_length"] + trans_length_to_term)
#
#         # termination vias
#         components_along_path_electrical = []
#         sections_electrical = []
#         _ = sections_electrical.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section for whatever reason (dummy section)
#
#         num_rows_via_contact_term = int(htr_connect_width / (via_size_contact + gap_via_contact)) - 2
#         offset_via_2 = num_rows_via_contact_term * (via_size_contact + gap_via_contact) / 2 - gap_via_contact
#         offset_sig = htr_closer_params["pad_center_gnd_width"] / 2 + htr_closer_params["pad_inner_gap_width"] + htr_closer_params["pad_sig_width"] - 7.9
#         for i in range(num_rows_via_contact_term):
#             components_along_path_electrical.append(
#                 ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT), spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#                                    offset=(offset_sig + i * (via_size_contact + gap_via_contact))))
#             components_along_path_electrical.append(
#                 ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT), spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#                                    offset=-(offset_sig + i * (via_size_contact + gap_via_contact))))
#             components_along_path_electrical.append(
#                 ComponentAlongPath(component=gf.c.via(size=(via_size_contact, via_size_contact), layer=CONTACT), spacing=via_size_contact + gap_via_contact, padding=via_size_contact + gap_via_contact,
#                                    offset=(-offset_via_2 + i * (via_size_contact + gap_via_contact))))
#
#         num_rows_via1_term = int(htr_connect_width / (via_size_1 + gap_via_1)) - 2
#         offset_via_1 = num_rows_via1_term * (via_size_1 + gap_via_1) / 2 - gap_via_1
#         # for i in range(num_rows_via1_term): #already placed by create_vias_inside_shape
#         #     components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_1, via_size_1), layer=VIA1), spacing=via_size_1 + gap_via_1, padding=via_size_1 + gap_via_1,
#         #                                                                offset=(offset_sig + i * (via_size_1 + gap_via_1))))
#         #     components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_1, via_size_1), layer=VIA1), spacing=via_size_1 + gap_via_1, padding=via_size_1 + gap_via_1,
#         #                                                                offset=-(offset_sig + i * (via_size_1 + gap_via_1))))
#         #     components_along_path_electrical.append(ComponentAlongPath(component=gf.c.via(size=(via_size_1, via_size_1), layer=VIA1), spacing=via_size_1 + gap_via_1, padding=via_size_top + gap_via_1,
#         #                                                                offset=(-offset_via_1 + i * (via_size_1 + gap_via_1))))
#
#         b = gf.Component("M1_term_pads")
#         # m1 pads
#         offset_m1 = offset_sig + htr_connect_width / 2 - (2 * via_size_1 + 2 * gap_via_1)
#         m1_pad = gf.components.rectangle(size=[htr_connect_length, htr_connect_width], layer=MT1)
#         _ = b << m1_pad
#         _.move([htr_center - htr_connect_length / 2, -htr_connect_width / 2 + offset_m1])
#         _ = b << m1_pad
#         _.move([htr_center - htr_connect_length / 2, -htr_connect_width / 2 - offset_m1])
#         _ = b << m1_pad
#         _.move([htr_center - htr_connect_length / 2, -htr_connect_width / 2])
#
#         b = gf.geometry.fillet(b, radius=params["htr_connect_fillet_radius"])
#         c_heater.add_polygon(b)
#         c_heater.add_polygon(b, layer=HTR)
#
#         x1_2 = gf.CrossSection(sections=sections_electrical, components_along_path=components_along_path_electrical)
#         p1_2 = gf.path.straight(length=htr_connect_length)
#         PS_e = gf.path.extrude(p1_2, x1_2)
#         _ = c_heater << PS_e
#         _.movex(htr_center - htr_connect_length / 2)  # params["pad_length"] + 2*params["S2S_length"] + params["trans_length"] + params["PS_length"] + trans_length_to_term + 15)
#
#
#         _ = c << c_heater
#         # _.movey(pad_t_center_gnd_width/2+pad_t_inner_gap_width+pad_t_sig_width/2)
#         # _ = c << c_heater
#         # _.movey(-(pad_t_center_gnd_width/2+pad_t_inner_gap_width+pad_t_sig_width/2))
#
#     PS_MT1_center_gnd_width = PS_center_gnd_width
#     PS_MT1_inner_gap_width = PS_inner_gap_width
#     PS_MT1_sig_width = PS_sig_width
#     PS_MT1_outer_gap_width = PS_outer_gap_width
#     PS_MT1_outer_gnd_width = PS_outer_gnd_width
#
#     #M1 layer
#     if not params["MT1_from_PS"]:
#         S2S_M1_center_gnd_width_local = S2S_center_gnd_width_local
#         S2S_M1_inner_gap_width_local = S2S_inner_gap_width_local
#
#         if params["DC_MT1"]:
#             PS_MT1_inner_gap_width = PS_MT1_inner_gap_width + PS_MT1_center_gnd_width / 2
#             PS_MT1_center_gnd_width = 0
#             S2S_M1_inner_gap_width_local = S2S_inner_gap_width_local + S2S_center_gnd_width_local / 2
#             S2S_M1_center_gnd_width_local = 0
#
#         if params["PS_taper"]:
#
#             pad_1, x1 = SGS_piece(0, pad_inner_gap_width + pad_center_gnd_width/2 + pad_sig_width/2-1, 2, pad_outer_gap_width + pad_sig_width/2-1 + pad_outer_gnd_width/2-1, 2, 0, MT1)
#             _ = c << pad_1
#             _.movex(pad_length)
#
#             # S2S 1
#             s2s_1, x2 = SGS_piece(S2S_M1_center_gnd_width_local, S2S_M1_inner_gap_width_local, S2S_sig_width_local,
#                                     S2S_outer_gap_width_local, S2S_outer_gnd_width_local, S2S_length, MT1)
#             _ = c << s2s_1
#             _.movex(trans_length + pad_length)
#
#             taper_MT1_1 = gf.components.taper_cross_section_linear(x1, x2, length=trans_length)
#             _ = c << taper_MT1_1
#             _.movex(pad_length)
#
#             ps, x3 = SGS_piece(PS_MT1_center_gnd_width,
#                                 PS_MT1_inner_gap_width,
#                                 PS_MT1_sig_width,
#                                 PS_MT1_outer_gap_width,
#                                 PS_MT1_outer_gnd_width,
#                                 PS_length-2*PS_trans_length, MT1)
#             ps_ref = c << ps
#             ps_ref.movex(trans_length + pad_length + S2S_length + PS_trans_length)
#
#             # print(PS_trans_length)
#             # print(type(PS_trans_length))
#             taper_PS1 = gf.components.taper_cross_section_linear(x2, x3, length=PS_trans_length)
#
#             _ = c << taper_PS1
#             _.movex(pad_length + trans_length + S2S_length)
#
#             # S2S 2
#             s2s_1, x4 = SGS_piece(S2S_M1_center_gnd_width_local, S2S_M1_inner_gap_width_local, S2S_sig_width_local,
#                                     S2S_outer_gap_width_local, S2S_outer_gnd_width_local, S2S_length, MT1)
#             _ = c << s2s_1
#             _.movex(trans_length + pad_length + S2S_length + PS_length)
#
#             taper_PS2 = gf.components.taper_cross_section_linear(x3, x4, length=PS_trans_length)
#             _ = c << taper_PS2
#             _.movex(pad_length + trans_length + S2S_length + PS_length - PS_trans_length)
#
#             if params["RF_out"]:
#                 pad_2, x5 = SGS_piece(0, pad_inner_gap_width + pad_center_gnd_width/2 + pad_sig_width/2-1, 2, pad_outer_gap_width + pad_sig_width/2-1 + pad_outer_gnd_width/2-1, 2, 0, MT1)
#                 _ = c << pad_2
#                 _.movex(pad_length + trans_length + 2*S2S_length + PS_length + trans_length_to_term)
#
#                 taper_MT1_2 = gf.components.taper_cross_section_linear(x4, x5, length=trans_length_to_term)
#                 _ = c << taper_MT1_2
#                 _.movex(pad_length + trans_length + 2*S2S_length + PS_length)
#
#
#             #VIAS IN tapered areas
#
#             def create_vias_inside_shape(params, P_shapely, inner_margin, outer_margin=1.3):
#                 #inner margin determines how thick the via layer is
#                 #outer margin determines space between vias and geometry edge
#                 #P_shapely = Polygon(list(zip(x_points, y_points)))
#                 P_shapely_shrink = P_shapely.buffer(inner_margin)
#                 P_shapely_place = shapely.difference(P_shapely, P_shapely_shrink) #get donut
#                 P_shapely_final = P_shapely_place
#                 if not params["RF_out"]: #remove end enclosing vias for open termination version
#                     a = P_shapely_place.bounds
#                     P_shapely_subtract = Polygon([(a[2] - abs(inner_margin), a[1]), (a[2] - abs(inner_margin), a[3]), (a[2], a[3]), (a[2], a[1])])
#                     P_shapely_final = shapely.difference(P_shapely_place, P_shapely_subtract)
#
#                 ref_Metals = gf.Component()
#                 ref_Metals.add_polygon(P_shapely_final, layer=MT1)
#                 Tr1 = [(ref_Metals.xmin - 5, ref_Metals.ymin - 5), (ref_Metals.xmin - 5, ref_Metals.ymax + 5), (ref_Metals.xmax + 5, ref_Metals.ymax + 5), (ref_Metals.xmax + 5, ref_Metals.ymin - 5)]
#                 ref_Metals.add_polygon(Tr1, layer=MT1_DUMMY_BLOCK)
#                 fill_size = [0.8, 0.8]
#
#                 return gf.fill_rectangle(
#                     ref_Metals,
#                     fill_size=fill_size,
#                     fill_layers=[VIA1],
#                     margin=outer_margin,
#                     fill_densities=[0.191],
#                     avoid_layers=[MT1_DUMMY_BLOCK],
#                     include_layers=[MT1]
#                 )
#
#             class ViaPath:
#                 def __init__(self, line_coor_x, line_coor_y, offset):
#                     self.line_coor_x = line_coor_x
#                     self.line_coor_y = line_coor_y
#                     self.offset = offset
#
#             if via_place_version == 2:
#                 inner_margin = -10
#                 #ref_Metals =
#                 d = gf.Component("dummy")
#                 d << c
#
#                 if not params["RF_out"]:
#                     s2s_out, x4 = SGS_piece(S2S_M1_center_gnd_width_local, S2S_M1_inner_gap_width_local, S2S_sig_width_local,
#                                               S2S_outer_gap_width_local, S2S_outer_gnd_width_local, abs(inner_margin), MT1)
#                     _ = d << s2s_out
#                     _.movex(trans_length + pad_length + S2S_length + PS_length+S2S_length) #append extra M1 to extend vias to proper length
#                 #d.show()
#                 MT1_poly_list = d.get_polygons(by_spec=MT1, as_shapely=True)
#                 MT1_poly = shapely.coverage_union_all(MT1_poly_list)
#                 # for i in MT1_poly_list:
#                 #     gf.Component()
#                 c << create_vias_inside_shape(params, MT1_poly, inner_margin=inner_margin, outer_margin=1.3)
#
#                 # components_along_path = []
#                 # sections_dummy = []
#                 # sections_dummy.append(gf.Section(width=0, layer=MT1, name="dummy"))  # CrossSection requires a section (dummmy) for multiple components
#                 #
#                 # num_rows_V1 = 9
#                 # offset_via_1 = -num_rows_V1 * (via_size_top + gap_via_top) / 2 + gap_via_top
#                 # offset_MT1_inner_sig_1 = S2S_center_gnd_width / 2 + S2S_inner_gap_width + (num_rows_V1 +2) * (via_size_top + gap_via_top) / 2 + 0.4
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(offset_MT1_inner_sig_1 + offset_via_1 + i * (via_size_top + gap_via_top))))
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(-(offset_MT1_inner_sig_1 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#                 #
#                 # offset_MT1_inner_sig_2 = S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width - (num_rows_V1 - 2) * (via_size_top + gap_via_top) + 0.545
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(offset_MT1_inner_sig_2 + offset_via_1 + i * (via_size_top + gap_via_top))))
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(-(offset_MT1_inner_sig_2 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#                 #
#                 # num_rows_V1 = 10
#                 # offset_MT1_outer_sig_1 = S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width + (num_rows_V1 + 1) * (via_size_top + gap_via_top) / 2 + 0.4 - 0.275
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(offset_MT1_outer_sig_1 + offset_via_1 + i * (via_size_top + gap_via_top))))
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(-(offset_MT1_outer_sig_1 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#                 #
#                 # offset_MT1_outer_sig_2 = S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width + S2S_outer_gnd_width - (num_rows_V1 - 3) * (via_size_top + gap_via_top) + 0.545 - 0.285
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(offset_MT1_outer_sig_2 + offset_via_1 + i * (via_size_top + gap_via_top))))
#                 # for i in range(num_rows_V1):
#                 #     components_along_path.append(
#                 #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                 #                            offset=(-(offset_MT1_outer_sig_2 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#                 #
#                 # x1 = gf.CrossSection(sections=sections_dummy,components_along_path=components_along_path)
#                 # p1 = gf.path.straight(length=abs(inner_margin)+1.3)  # , bend=gf.path.arc, radius=50
#                 # PS = gf.path.extrude(p1, x1)
#                 # _ = c << PS
#                 # _.movex(pad_length+trans_length+S2S_length+PS_length+S2S_length-abs(inner_margin)-1.3)
#
#             viaPathArray=[]
#             sections_electrical_extend = []
#             s = sections_electrical_extend.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section (dummmy) for multiple components
#             path_points_array = []
#             path_points_narrow_array = []
#             num_rows_V1 = 9
#             num_rows_V1_narrow = 2
#             offset_via_1 = -num_rows_V1 * (via_size_top + gap_via_top) / 2 + gap_via_top
#             offset_via_1_narrow = -num_rows_V1_narrow * (via_size_top + gap_via_top) / 2 + gap_via_top
#
#             #original points defined in top left corner of GSGSG
#             #outer gnd taper top
#             x_points = [pad_length + 22,
#                         pad_length + trans_length]
#             y_points = [pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width + pad_outer_gnd_width / 2 +5.75,
#                         S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width + S2S_outer_gnd_width ]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=-1))
#
#             #outer gnd taper bottom
#             x_points = [pad_length + 22,
#                         pad_length + trans_length]
#             y_points = [ pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width+ pad_outer_gnd_width/2  -13 ,
#                         S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width ]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#             if via_place_version == 1:
#                 #shapely ver for tips
#                 x_points = [pad_length,
#                             pad_length + 22 + 9.2,
#                             pad_length + 22 + 9.2,
#                             pad_length]
#                 y_points = [pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width+ pad_outer_gnd_width/2 +1,
#                             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width+ pad_outer_gnd_width/2 +6.3,
#                             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width+ pad_outer_gnd_width/2 +1-25.3+6,
#                             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width+ pad_outer_gnd_width/2 -1]
#
#                 c << create_vias_inside_shape(params, Polygon(list(zip(x_points, y_points))), inner_margin=-9)
#                 y_bot = [-y for y in y_points]
#                 c << create_vias_inside_shape(params, Polygon(list(zip(x_points, y_bot))), inner_margin=-9)
#
#             #outer gnd s2s bottom
#             x_points = [pad_length + trans_length,
#                         pad_length + trans_length + S2S_length]
#             y_points = [S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width ,
#                         S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#             #signal s2s top
#             x_points = [pad_length + trans_length,
#                         pad_length + trans_length + S2S_length]
#             y_points = [S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width ,
#                         S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=-1))
#
#             #outer gnd s2s bottom termination side
#             x_points = [pad_length + trans_length + S2S_length + PS_length + 0.5,
#                         pad_length + trans_length + S2S_length + PS_length + S2S_length]
#             y_points = [S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width ,
#                         S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width ]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#             # signal s2s top termination side
#             x_points = [pad_length + trans_length + S2S_length + PS_length + 1.5,
#                         pad_length + trans_length + S2S_length + PS_length + S2S_length]
#             y_points = [S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width,
#                         S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width]
#             viaPathArray.append(ViaPath(line_coor_x=x_points, line_coor_y=y_points, offset=-1))
#
#             #outer gnd ps taper bottom
#             x_points = [pad_length + trans_length + S2S_length + 0.5,
#                         pad_length + trans_length + S2S_length + PS_trans_length]
#             y_points = [S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width ,
#                         PS_center_gnd_width/2 + PS_inner_gap_width + PS_sig_width + PS_outer_gap_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#             #signal ps taper top
#             x_points = [pad_length + trans_length + S2S_length + 0.5,
#                         pad_length + trans_length + S2S_length + PS_trans_length+1.3]
#             y_points = [S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width ,
#                         PS_center_gnd_width/2 + PS_inner_gap_width + PS_sig_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=-1))
#
#             #signal ps taper top termination side
#             x_points = [pad_length + trans_length + S2S_length + PS_length -PS_trans_length,
#                         pad_length + trans_length + S2S_length + PS_length+1]
#             y_points = [PS_center_gnd_width/2 + PS_inner_gap_width + PS_sig_width,
#                 S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width]
#
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=-1))
#
#             # outer gnd ps taper bottom termination side
#             x_points = [pad_length + trans_length + S2S_length + PS_length - PS_trans_length,
#                         pad_length + trans_length + S2S_length + PS_length]
#             y_points = [PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width + PS_outer_gap_width,
#                         S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#             # outer gnd ps short
#             x_points = [pad_length + trans_length + S2S_length + PS_trans_length,
#                         pad_length + trans_length + S2S_length + PS_length - PS_trans_length]
#             y_points = [PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width + PS_outer_gap_width,
#                         PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width + PS_outer_gap_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#             # outer gnd ps
#             x_points = [pad_length + trans_length,
#                         pad_length + trans_length + 2*S2S_length + PS_length - 0.5]
#             y_points = [PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width + PS_outer_gap_width + PS_outer_gnd_width,
#                         PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width + PS_outer_gap_width + PS_outer_gnd_width]
#             viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=-1))
#
#             # # inner signal ps outer
#             # x_points = [pad_length + trans_length,
#             #             pad_length + trans_length + 2 * S2S_length + PS_length - 0.5]
#             # y_points = [PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width,
#             #             PS_center_gnd_width / 2 + PS_inner_gap_width + PS_sig_width ]
#             # viaPathArray.append(ViaPath(line_coor_x=x_points, line_coor_y=y_points, offset=-1))
#             #
#             # # inner signal ps inner
#             # x_points = [pad_length + trans_length,
#             #             pad_length + trans_length + 2 * S2S_length + PS_length - 0.5]
#             # y_points = [PS_center_gnd_width / 2 + PS_inner_gap_width,
#             #             PS_center_gnd_width / 2 + PS_inner_gap_width]
#             # viaPathArray.append(ViaPath(line_coor_x=x_points, line_coor_y=y_points, offset=1))
#
#             # #signal taper upper
#             # x_points = [pad_length + trans_length/4,
#             #             pad_length + trans_length-0.5]
#             # y_points = [pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width/2 - (pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width/2 - (S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width))/4 ,
#             #             PS_MT1_center_gnd_width/2 + PS_MT1_inner_gap_width + PS_MT1_sig_width]
#             # viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=-1))
#             #
#             # # signal taper lower
#             # x_points = [pad_length + 0.46*trans_length,
#             #             pad_length + trans_length -1]
#             # y_points = [pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width/2 - (pad_center_gnd_width/2 + pad_inner_gap_width + pad_sig_width/2 - 0.93*(S2S_center_gnd_width/2 + S2S_inner_gap_width + S2S_sig_width)),
#             #             PS_MT1_center_gnd_width/2 + PS_MT1_inner_gap_width ]
#             # viaPathArray.append(ViaPath(line_coor_x = x_points, line_coor_y = y_points, offset=1))
#
#
#             #orignal placement
#             # # signal taper, narrow
#             # x_points = [pad_length + 1,
#             #             pad_length + trans_length / 4]
#             # y_points = [pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - 0.15,
#             #             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - (
#             #                         pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - (S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width)) / 4 - 3.5]
#             # path_points_narrow_array.append([x_points, y_points])
#
#             #fill placement
#             # add vias
#             # ref_Metals = gf.Component("ref_metal")
#             # x_points = [pad_length + 1,
#             #             pad_length + trans_length / 4,
#             #             pad_length + trans_length / 4,
#             #             pad_length + 1]
#             # y_points = [pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 +1,
#             #             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - (pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - (S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width)) / 4,
#             #             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - (pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - (S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width)) / 4 - 10,
#             #             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 - 1]
#             #
#             # ref_Metals.add_polygon([x_points, y_points], layer=MT1)
#             # Tr1 = [(ref_Metals.xmin-5, ref_Metals.ymin-5), (ref_Metals.xmin-5, ref_Metals.ymax+5), (ref_Metals.xmax+5, ref_Metals.ymax+5), (ref_Metals.xmax+5, ref_Metals.ymin-5)]
#             # ref_Metals.add_polygon(Tr1, layer=MT1_DUMMY_BLOCK)
#
#             #
#             #
#             # fill_size = [0.8, 0.8]
#             # c << gf.fill_rectangle(
#             #     ref_Metals,
#             #     fill_size=fill_size,
#             #     fill_layers=[VIA1],
#             #     margin=1,
#             #     fill_densities=[0.191],
#             #     avoid_layers=[MT1_DUMMY_BLOCK],
#             #     include_layers=[MT1]
#             # )
#             #
#             # c << ref_Metals
#
#             if via_place_version == 1:
#                 #shapely ver
#                 x_points = [pad_length,
#                             pad_length + trans_length+7.5,
#                             pad_length + trans_length+7.5,
#                             pad_length]
#                 y_points = [pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 +1,
#                             S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width-1,
#                             S2S_center_gnd_width / 2 + S2S_inner_gap_width-1.5,
#                             pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2 -1]
#
#                 c << create_vias_inside_shape(params, Polygon(list(zip(x_points, y_points))), inner_margin=-8)
#                 y_bot = [-y for y in y_points]
#                 c << create_vias_inside_shape(params, Polygon(list(zip(x_points, y_bot))), inner_margin=-8)
#
#
#
#                 #legacy via placement system
#                 for i in path_points_array:
#
#                     x_points = i[0]
#                     y_points = i[1]
#                     rotation = math.asin((y_points[1] - y_points[0]) / (x_points[1]-x_points[0]))
#                     rotation = math.degrees(rotation)
#                     # print("x_points: " + str(x_points))
#                     # print("y_points: " + str(y_points))
#                     # print("rotation: " + str(rotation))
#                     #temporary hack fix until bug is fixed
#                     if x_points[0] == 90:
#                         rotation = 10.9
#                     if x_points[0] == 70:
#                         rotation = -29.5
#                     if x_points[0] == 361:
#                         rotation = 40.33 - 7.3
#                     if x_points[0] == 860:
#                         rotation = -40.41 + 7.4
#                     if x_points[0] == 210:
#                         rotation = -12.65
#
#                     components_along_path = []
#                     for j in range(num_rows_V1):
#                         components_along_path.append(
#                             ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1).rotate(-rotation), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                                offset=(offset_via_1 + j * (via_size_top + gap_via_top))))
#                     x1 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path)
#
#
#                     points = list(zip(x_points, y_points))
#                     p1 = gf.path.smooth(points) #, bend=gf.path.arc, radius=50
#                     PS = gf.path.extrude(p1, x1)
#                     _ = c << PS
#
#                     #repeat for flipped y
#                     y_points = [-y for y in y_points]
#                     rotation = math.asin((y_points[1] - y_points[0]) / (x_points[1] - x_points[0]))
#                     rotation = math.degrees(rotation)
#                     if x_points[0] == 90:
#                         rotation = -10.9
#                     if x_points[0] == 70:
#                         rotation = 29.5
#                     if x_points[0] == 361:
#                         rotation = -40.33 + 7.3
#                     if x_points[0] == 860:
#                         rotation = 40.41 - 7.4
#                     if x_points[0] == 210:
#                         rotation = 12.65
#                     components_along_path = []
#                     for j in range(num_rows_V1):
#                         components_along_path.append(
#                             ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1).rotate(-rotation), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                                offset=(offset_via_1 + j * (via_size_top + gap_via_top))))
#                     x1 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path)
#
#                     points = list(zip(x_points, y_points))
#                     p1 = gf.path.smooth(points)
#                     PS = gf.path.extrude(p1, x1)
#                     _ = c << PS
#
#                     #repeat for flip x side
#
#
#                 for i in path_points_narrow_array:
#
#                     x_points = i[0]
#                     y_points = i[1]
#                     rotation = math.asin((y_points[1] - y_points[0]) / (x_points[1]-x_points[0]))
#                     rotation = math.degrees(rotation)
#                     components_along_path = []
#                     for j in range(num_rows_V1_narrow):
#                         components_along_path.append(
#                             ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1).rotate(-rotation), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                                offset=(offset_via_1_narrow + j * (via_size_top + gap_via_top))))
#                     x1 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path)
#
#
#                     points = list(zip(x_points, y_points))
#                     p1 = gf.path.smooth(points)
#                     PS = gf.path.extrude(p1, x1)
#                     _ = c << PS
#
#                     #repeat for flipped y
#                     y_points = [-y for y in y_points]
#                     rotation = math.asin((y_points[1] - y_points[0]) / (x_points[1] - x_points[0]))
#                     rotation = math.degrees(rotation)
#                     components_along_path = []
#                     for j in range(num_rows_V1_narrow):
#                         components_along_path.append(
#                             ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1).rotate(-rotation), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                                offset=(offset_via_1_narrow + j * (via_size_top + gap_via_top))))
#                     x1 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path)
#
#                     points = list(zip(x_points, y_points))
#                     p1 = gf.path.smooth(points)
#                     PS = gf.path.extrude(p1, x1)
#                     _ = c << PS
#
#                     #repeat for flip x side
#
#
#                 for i in viaPathArray:
#                     c << PEO_custom_via_from_line(line_coor_x=i.line_coor_x, line_coor_y=i.line_coor_y, via_vdistfromline=i.offset)
#                     y_bot = [-y for y in i.line_coor_y]
#                     c << PEO_custom_via_from_line(line_coor_x=i.line_coor_x, line_coor_y=y_bot, via_vdistfromline=-i.offset)
#
#
#         else:
#             ps, _ = SGS_piece(PS_MT1_center_gnd_width,
#                               PS_MT1_inner_gap_width,
#                               PS_MT1_sig_width,
#                               PS_MT1_outer_gap_width,
#                               PS_MT1_outer_gnd_width,
#                               PS_length, MT1)
#             ps_ref = c << ps
#             ps_ref.movex(trans_length + pad_length + S2S_length)
#
#
#     if params["DC_MT1"]: #central M1 structures
#         if params["s2s_type"] == "adiabatic":
#             ps, x3 = SGS_piece(4, #center gnd
#                                  0,
#                                  0,
#                                  0,
#                                  0,
#                                  PS_length+ 2*(params["S2S_ADIA_L3"]+params["S2S_ADIA_L2"]), MT1)
#             ps_ref = c << ps
#             ps_ref.movex(trans_length + pad_length +4)
#
#             ps, x3 = SGS_piece(0,  #dc rails
#                                  PS_center_gnd_width / 2 - 5,
#                                  4,
#                                  0,
#                                  0,
#                                  PS_length + 2*(params["S2S_ADIA_L3"]+params["S2S_ADIA_L2"]) +2*8, MT1)
#             ps_ref = c << ps
#             ps_ref.movex(trans_length + pad_length - 4)# + S2S_length - 12)
#
#             s2s, x3 = SGS_piece(PS_center_gnd_width - 2, #horizontal bar piece, top
#                                   0,
#                                   0,
#                                   0,
#                                   0,
#                                   4, MT1)
#             s2s_ref = c << s2s
#             s2s_ref.movex(trans_length + pad_length - 8)
#
#             s2s, x3 = SGS_piece(PS_center_gnd_width - 2,  #horizontal bar piece, bottom
#                                   0,
#                                   0,
#                                   0,
#                                   0,
#                                   4, MT1)
#             s2s_ref = c << s2s
#             s2s_ref.movex(pad_length + trans_length + S2S_length + PS_length + S2S_length +4)
#
#             pad, x3 = SGS_piece(4, #DC probe extension, top
#                                   0,
#                                   0,
#                                   0,
#                                   0,
#                                   pad_length + trans_length + 10 - 8, MT1)
#             pad_ref = c << pad
#             pad_ref.movex(-10)
#
#             pad, x3 = SGS_piece(4,    #DC probe extension, bottom
#                                   0,
#                                   0,
#                                   0,
#                                   0,
#                                   10 , MT1)
#             pad_ref = c << pad
#             pad_ref.movex(pad_length + trans_length + PS_length + 2*(params["S2S_ADIA_L3"]+params["S2S_ADIA_L2"]) + 16)
#
#             # central column of vias + ALL vias
#             components_along_path_PS = []
#             sections_electrical_extend = []
#             s = sections_electrical_extend.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section (dummmy) for multiple components
#
#             num_rows_V1 = 4
#             offset_via_1 = -num_rows_V1 * (via_size_top + gap_via_top) / 2 + gap_via_top
#             for i in range(num_rows_V1):
#                 components_along_path_PS.append(
#                     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                        offset=(offset_via_1 + i * (via_size_top + gap_via_top))))
#
#             x3 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path_PS)
#
#             if params["s2s_type"] == "adiabatic":
#                 p3 = gf.path.straight(length=PS_length+2*(params["S2S_ADIA_L3"]+params["S2S_ADIA_L2"]))
#                 PS = gf.path.extrude(p3, x3)
#                 _ = c << PS
#                 _.movex(params["pad_length"] + params["trans_length"] + S2S_length - (params["S2S_ADIA_L3"]+params["S2S_ADIA_L2"]))
#             else:
#                 p3 = gf.path.straight(length=PS_length)
#                 PS = gf.path.extrude(p3, x3)
#                 _ = c << PS
#                 _.movex(params["pad_length"] + params["trans_length"] + S2S_length)
#         else:
#             ps, x3 = SGS_piece(4,
#                                  0,
#                                  0,
#                                  0,
#                                  0,
#                                  PS_length, MT1)
#             ps_ref = c << ps
#             ps_ref.movex(trans_length + pad_length + S2S_length)
#
#             ps, x3 = SGS_piece(0,
#                                  PS_center_gnd_width/2-5,
#                                  4,
#                                  0,
#                                  0,
#                                  PS_length + 2*12, MT1)
#             ps_ref = c << ps
#             ps_ref.movex(trans_length + pad_length + S2S_length-12)
#
#             s2s, x3 = SGS_piece(PS_center_gnd_width-2,
#                                  0,
#                                  0,
#                                  0,
#                                  0,
#                                  4, MT1)
#             s2s_ref = c << s2s
#             s2s_ref.movex(trans_length + pad_length + S2S_length-12)
#
#             s2s, x3 = SGS_piece(PS_center_gnd_width-2,
#                                  0,
#                                  0,
#                                  0,
#                                  0,
#                                  4, MT1)
#             s2s_ref = c << s2s
#             s2s_ref.movex(pad_length + trans_length + S2S_length + PS_length +12 - 4)
#
#             pad, x3 = SGS_piece(4,
#                                   0,
#                                   0,
#                                   0,
#                                   0,
#                                   pad_length + trans_length + 10 + S2S_length-12, MT1)
#             pad_ref = c << pad
#             pad_ref.movex(-10)
#
#             pad, x3 = SGS_piece(4,
#                                   0,
#                                   0,
#                                   0,
#                                   0,
#                                    10+  S2S_length-12, MT1)
#             pad_ref = c << pad
#             pad_ref.movex(pad_length + trans_length + S2S_length + PS_length + 12)
#
#             # central column of vias + ALL vias
#             components_along_path_PS = []
#             sections_electrical_extend = []
#             s = sections_electrical_extend.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section (dummmy) for multiple components
#
#             num_rows_V1 = 4
#             offset_via_1 = -num_rows_V1 * (via_size_top + gap_via_top) / 2 + gap_via_top
#             for i in range(num_rows_V1):
#                 components_along_path_PS.append(
#                     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                        offset=(offset_via_1 + i * (via_size_top + gap_via_top))))
#
#             x3 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path_PS)
#             p3 = gf.path.straight(length=PS_length)
#             PS = gf.path.extrude(p3, x3)
#             _ = c << PS
#             _.movex(params["pad_length"] + params["trans_length"] + S2S_length)
#
#
#         sections_electrical_extend = []
#         components_along_path = []
#         components_along_path_PS_center = []
#         s = sections_electrical_extend.append(gf.Section(width=0, layer=MT1, name="MT1_1"))  # CrossSection requires a section (dummmy) for multiple components
#         if via_place_version == 1:
#             num_rows_V1 = 10
#             offset_MT1_inner_sig_1 = PS_MT1_center_gnd_width / 2 + PS_MT1_inner_gap_width + (num_rows_V1-4) * (via_size_top + gap_via_top)/2
#             for i in range(num_rows_V1):
#                 components_along_path.append(
#                     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                        offset=(offset_MT1_inner_sig_1 + offset_via_1 + i * (via_size_top + gap_via_top))))
#             for i in range(num_rows_V1):
#                 components_along_path.append(
#                     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                        offset=(-(offset_MT1_inner_sig_1 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#
#
#             offset_MT1_inner_sig_2 = PS_MT1_center_gnd_width/2 + PS_MT1_inner_gap_width + PS_MT1_sig_width - (num_rows_V1-1) * (via_size_top + gap_via_top)
#             for i in range(num_rows_V1):
#                 components_along_path_PS_center.append(
#                     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                        offset=(offset_MT1_inner_sig_2 + offset_via_1 + i * (via_size_top + gap_via_top))))
#             for i in range(num_rows_V1):
#                 components_along_path_PS_center.append(
#                     ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#                                        offset=(-(offset_MT1_inner_sig_2 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#
#         # offset_MT1_outer_gnd_1_S2S = S2S_center_gnd_width / 2 + S2S_inner_gap_width + S2S_sig_width + S2S_outer_gap_width + (num_rows_V1 - 1) * (via_size_top + gap_via_top) / 2
#         # for i in range(num_rows_V1):
#         #     components_along_path_S2S.append(
#         #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#         #                            offset=(offset_MT1_outer_gnd_1_S2S + offset_via_1 + i * (via_size_top + gap_via_top))))
#         # for i in range(num_rows_V1):
#         #     components_along_path_S2S.append(
#         #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#         #                            offset=(-(offset_MT1_outer_gnd_1_S2S + offset_via_1 + i * (via_size_top + gap_via_top)))))
#
#         # offset_MT1_outer_gnd_1 = PS_MT1_center_gnd_width / 2 + PS_MT1_inner_gap_width + PS_MT1_sig_width + PS_MT1_outer_gap_width + (num_rows_V1-4) * (via_size_top + gap_via_top)/2
#         # for i in range(num_rows_V1):
#         #     components_along_path_PS_center.append(
#         #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#         #                            offset=(offset_MT1_outer_gnd_1 + offset_via_1 + i * (via_size_top + gap_via_top))))
#         # for i in range(num_rows_V1):
#         #     components_along_path_PS_center.append(
#         #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#         #                            offset=(-(offset_MT1_outer_gnd_1 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#
#         # offset_MT1_outer_gnd_2 = PS_MT1_center_gnd_width / 2 + PS_MT1_inner_gap_width + PS_MT1_sig_width + PS_MT1_outer_gap_width + PS_MT1_outer_gnd_width - (num_rows_V1-1) * (via_size_top + gap_via_top)
#         # for i in range(num_rows_V1):
#         #     components_along_path.append(
#         #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#         #                            offset=(offset_MT1_outer_gnd_2 + offset_via_1 + i * (via_size_top + gap_via_top))))
#         # for i in range(num_rows_V1):
#         #     components_along_path.append(
#         #         ComponentAlongPath(component=gf.c.via(size=(via_size_top, via_size_top), layer=VIA1), spacing=via_size_top + gap_via_top, padding=via_size_top + gap_via_top,
#         #                            offset=(-(offset_MT1_outer_gnd_2 + offset_via_1 + i * (via_size_top + gap_via_top)))))
#
#         x1 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path)
#         p1 = gf.path.straight(length=PS_length + 2*S2S_length + 1)
#         PS = gf.path.extrude(p1, x1)
#         _ = c << PS
#         _.movex(params["pad_length"] + params["trans_length"] -1)
#
#         x2 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path_PS_center)
#         p2 = gf.path.straight(length=PS_length - 2*PS_trans_length)
#         PS = gf.path.extrude(p2, x2)
#         _ = c << PS
#         _.movex(params["pad_length"] + params["trans_length"]+ PS_trans_length + S2S_length )
#
#
#
#         # x3 = gf.CrossSection(sections=sections_electrical_extend, components_along_path=components_along_path_S2S)
#         # p3 = gf.path.straight(length=S2S_length)
#         # PS = gf.path.extrude(p3, x3)
#         # _ = c << PS
#         # _.movex(params["pad_length"] + params["trans_length"] )
#         # _ = c << PS
#         # _.movex(params["pad_length"] + params["trans_length"]+S2S_length + PS_length )
#
#
#
#     #M2 layer
#     if not params["RF_out"]: #gnd addition for Crealights frame
#         pad_in, x1 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width + pad_sig_width, 0, pad_outer_gap_width,
#                                  pad_outer_gnd_width, 60+15, layer_MT2)
#         _ = c << pad_in
#         _.movex(-60/2 - 15)
#
#         # pad_in, x1 = SGS_piece(pad_center_gnd_width + 2*pad_inner_gap_width + 2*pad_sig_width + 2*pad_outer_gap_width  + pad_outer_gnd_width, 0, 0,
#         #                          0, 0, 15, layer_MT2)
#         # _ = c << pad_in
#         # _.movex(-(20+15))
#
#     if "htr_bridge_place" not in params:
#         params["htr_bridge_place"] = False
#     if params["htr_bridge_place"] and params["DC_MT1"]: #DC_MT1 version
#         bridges = htr_bridges_DC_MT1(params)
#         _ = c << bridges
#         _.rotate(180)
#         _.movex(pad_length+trans_length - 10) #should be -10 only if "s2s_O_len_OXOP" == 50
#
#         if params["RF_out"]:
#             _ = c << bridges
#             _.movex(pad_length + trans_length + S2S_length + PS_length + 50+ 10)
#
#
#     elif params["htr_bridge_place"]:
#         if params["s2s_type"] == "adiabatic":
#             bridges = htr_bridges(params, offset=S2S_length/2+10)
#             _ = c << bridges
#             _.rotate(180)
#             _.movex(pad_length + trans_length-10)
#             if params["RF_out"]:
#                 _ = c << bridges
#                 _.movex(pad_length + trans_length + S2S_length + PS_length + S2S_length+10)
#
#         elif params["s2s_O_len_OXOP"] > 0:
#             bridges = htr_bridges(params, offset=params["s2s_O_len_OXOP"]+10)
#             _ = c << bridges
#             _.rotate(180)
#             _.movex(pad_length + trans_length+S2S_length/2-params["s2s_O_len_OXOP"])
#             if params["RF_out"]:
#                 _ = c << bridges
#                 _.movex(pad_length + trans_length + S2S_length + PS_length + S2S_length/2+params["s2s_O_len_OXOP"])
#
#         else:
#             bridges = htr_bridges(params)
#             _ = c << bridges
#             _.rotate(180)
#             _.movex(pad_length+trans_length+S2S_length/2)
#             if params["RF_out"]:
#                 _ = c << bridges
#                 _.movex(pad_length + trans_length + S2S_length + PS_length +S2S_length/2)
#
#
#     # input contact pads
#     pad_in, x1 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width,
#                              pad_outer_gnd_width, pad_length/2, layer_MT2)
#     _ = c << pad_in
#     _.movex(pad_length/2)
#     pad_in, x1 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width,
#                              pad_outer_gnd_width, pad_length, layer_MT2)
#     pad_in = gf.geometry.fillet(pad_in, radius=13.7)
#     c.add_polygon(pad_in)
#
#     if params["RF_out"]: #add pad openings back in
#         pad_in_PAD, x150 = SGS_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10,
#                                        pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, layer_PAD)
#         pad_in_PAD = gf.geometry.fillet(pad_in_PAD, radius=13.7)
#         d = gf.Component("pad_in_PAD")
#         d.add_polygon(pad_in_PAD)
#         _ = c << d
#         _.movex(5)
#
#     # S2S 1
#     s2s_1, x2 = SGS_piece(S2S_center_gnd_width_local, S2S_inner_gap_width_local, S2S_sig_width_local,
#                             S2S_outer_gap_width_local, S2S_outer_gnd_width_local, S2S_length, layer_MT2)
#     _ = c << s2s_1
#     _.movex(trans_length + pad_length)
#
#     # phase shifter region
#     if params["PS_taper"]:
#         PS, x3 = SGS_piece(PS_center_gnd_width, PS_inner_gap_width, PS_sig_width, PS_outer_gap_width,
#                              PS_outer_gnd_width, PS_length - 2*PS_trans_length, layer_MT2)
#         _ = c << PS
#         _.movex(trans_length + pad_length + S2S_length + PS_trans_length)
#     else:
#         PS, x3 = SGS_piece(PS_center_gnd_width, PS_inner_gap_width, PS_sig_width, PS_outer_gap_width,
#                                  PS_outer_gnd_width, PS_length, layer_MT2)
#         _ = c << PS
#         _.movex(trans_length + pad_length + S2S_length)
#
#     # transition 1
#     taper1 = gf.components.taper_cross_section_linear(x1, x2, length=trans_length)
#     _ = c << taper1
#     _.movex(pad_length)
#
#     if params["PS_taper"]:
#         taper_PS1 = gf.components.taper_cross_section_linear(x2, x3, length=PS_trans_length)
#         _ = c << taper_PS1
#         _.movex(pad_length + trans_length + S2S_length)
#
#
#
#     # S2S 2
#     s2s_1, x4 = SGS_piece(S2S_center_gnd_width_local, S2S_inner_gap_width_local, S2S_sig_width_local,
#                             S2S_outer_gap_width_local, S2S_outer_gnd_width_local, S2S_length, layer_MT2)
#     _ = c << s2s_1
#     _.movex(trans_length + pad_length + S2S_length + PS_length)
#
#     # transition 2
#     if params["PS_taper"]:
#         taper_PS2 = gf.components.taper_cross_section_linear(x3, x4, length=PS_trans_length)
#         _ = c << taper_PS2
#         _.movex(pad_length + trans_length + S2S_length + PS_length - PS_trans_length)
#
#
#     if not params["RF_out"]: #gnd extension to fit frame
#         pad_t_out, x5 = SGS_piece(0, 195/2, 175, 0,
#                                     0, 650, layer_MT2)
#         _ = c << pad_t_out
#         _.movex(pad_length +trans_length+ 2*S2S_length + PS_length)
#
#
#     if params["RF_out"]:
#         # output contact pads, transition
#         d = gf.Component("pad_out")
#         pad_t_out, x5 = SGS_piece(pad_t_center_gnd_width, pad_t_inner_gap_width, pad_t_sig_width, pad_t_outer_gap_width,
#                                   pad_t_outer_gnd_width, pad_t_length/2, layer_MT2)
#         _ = d << pad_t_out
#         #_.movex()
#         pad_t_out, x5 = SGS_piece(pad_t_center_gnd_width, pad_t_inner_gap_width, pad_t_sig_width, pad_t_outer_gap_width,
#                                     pad_t_outer_gnd_width, pad_t_length, layer_MT2)
#         pad_t_out = gf.geometry.fillet(pad_t_out, radius=13.7)
#         d.add_polygon(pad_t_out)
#
#         e = gf.Component("out_pad_pad")
#         pad_t_out_PAD, x150 = SGS_piece(pad_t_center_gnd_width - 10, pad_t_inner_gap_width + 10, pad_t_sig_width - 10,
#                                         pad_t_outer_gap_width + 10, pad_t_outer_gnd_width - 10, pad_t_length - 10, layer_PAD)
#         #_ = c << pad_t_out_PAD
#
#         pad_t_out_PAD = gf.geometry.fillet(pad_t_out_PAD, radius=13.7)
#         e.add_polygon(pad_t_out_PAD, layer=PAD)
#         _ = d << e
#         _.movex(5)
#         #_.movex(5 + trans_length + pad_length + S2S_length * 2 + PS_length + trans_length_to_term)
#
#         _ = c << d
#         _.movex(trans_length + pad_length + 2*S2S_length + PS_length + trans_length_to_term)
#
#         taper2 = gf.components.taper_cross_section_linear(x4, x5, length=trans_length_to_term)
#         _ = c << taper2
#         _.movex(pad_length + trans_length + PS_length + 2 * S2S_length)
#
#
#     #Crealights frame ground connections
#     if "MT2_connect_left" not in params:
#         params["MT2_connect_left"] = "none"
#
#     if params["MT2_connect_left"] == "fine":
#         MT2_cnct_fine = gf.import_gds("frame_M2_Fine_Pitch.gds", with_metadata=False)
#         _ = c <<MT2_cnct_fine
#         _.rotate(-90)
#         _.move((30, 312.5))
#
#     if params["MT2_connect_left"] == "regular":
#         MT2_cnct_regular = gf.import_gds("frame_M2_Regular_Pitch.gds", with_metadata=False)
#         _ = c << MT2_cnct_regular
#         _.rotate(-90)
#         _.move((30, 312.5))
#
#     # dummy fill block
#     MT_dummy_margin = params["MT_dummy_margin"]
#     if params["RF_out"]:
#         MT_dummy_width = params["pad_center_gnd_width"] + 2 * params["pad_inner_gap_width"] + 2 * params["pad_sig_width"] + 2 * params["pad_outer_gap_width"] + 2 * params["pad_outer_gnd_width"]
#         MT_dummy_length = params["pad_length"] + params["trans_length"] + 2 * params["S2S_length"] + PS_length + trans_length_to_term + pad_t_length
#     else:
#         MT_dummy_width = params["PS_center_gnd_width"] + 2 * params["PS_inner_gap_width"] + 2 * params["PS_sig_width"] + 2 * params["PS_outer_gap_width"] + 2 * params["PS_outer_gnd_width"] #-  2*MT_dummy_margin
#         MT_dummy_length = params["pad_length"] + params["trans_length"] + 2 * params["S2S_length"] + PS_length
#
#
#     for i in [MT1_DUMMY_BLOCK, MT2_DUMMY_BLOCK]:  # , MT3_DUMMY_BLOCK]:
#         _ = c << gf.components.rectangle(size=(MT_dummy_length + 2 * MT_dummy_margin, MT_dummy_width + 2 * MT_dummy_margin), layer=i)
#         _.move((-MT_dummy_margin, -MT_dummy_margin - MT_dummy_width / 2))
#
#
#     #DETCH ditch to contain Fenglass
#     trench, x150 = SGS_piece(0, 0, 0,
#                                PS_center_gnd_width/2+PS_inner_gap_width+PS_sig_width+PS_outer_gap_width+PS_outer_gnd_width+140, 10, PS_length+2*S2S_length,  DETCH)
#     _ = c << trench
#     _.movex(pad_length+trans_length)
#
#
#     c.add_port(
#         name="e_low",
#         center=(pad_length + trans_length + S2S_length,
#                 -PS_center_gnd_width / 2 - PS_inner_gap_width / 2),
#         width=1,
#         orientation=0,
#         layer=MT2,
#         port_type="electrical",
#     )
#
#     c.add_port(
#         name="e_up",
#         center=(pad_length + trans_length + S2S_length,
#                 PS_center_gnd_width / 2 + PS_inner_gap_width / 2),
#         width=1,
#         orientation=0,
#         layer=MT2,
#         port_type="electrical",
#     )
#
#     return c


@gf.cell
def DC_pads(DC_pad_size_x=None, DC_pad_size_y=None, DC_pad_gap=30, params=None) -> Component:

    layer_MT2 = MT2
    layer_PAD = PAD

    # Handle both calling styles
    if params is not None:
        DC_pad_size_x = params["DC_pad_size_x"]
        DC_pad_size_y = params["DC_pad_size_y"]
        # Convert layer integers to tuples if needed
        layer_MT2_val = params.get("layer_MT2", 125)
        layer_PAD_val = params.get("layer_PAD", 150)
        layer_MT2 = (layer_MT2_val, 0) if isinstance(layer_MT2_val, int) else layer_MT2_val
        layer_PAD = (layer_PAD_val, 0) if isinstance(layer_PAD_val, int) else layer_PAD_val
    else:
        # Use passed arguments or defaults
        if DC_pad_size_x is None or DC_pad_size_y is None:
            raise ValueError("Either provide params dict or both DC_pad_size_x and DC_pad_size_y")


    c = gf.Component()

    # Metal pads
    pad1 = c << gf.components.pad(
        size=[DC_pad_size_x, DC_pad_size_y],
        layer=layer_MT2,
        port_inclusion=1,
        port_orientation=90
    )
    pad2 = c << gf.components.pad(
        size=[DC_pad_size_x, DC_pad_size_y],
        layer=layer_MT2,
        port_inclusion=1,
        port_orientation=90
    )

    # Open overlay pads
    open1 = c << gf.components.pad(
        size=[DC_pad_size_x - 10, DC_pad_size_y - 10],
        layer=layer_PAD
    )
    open2 = c << gf.components.pad(
        size=[DC_pad_size_x - 10, DC_pad_size_y - 10],
        layer=layer_PAD
    )

    # Position second pad
    pad2.movex(DC_pad_size_x + DC_pad_gap)
    open2.movex(DC_pad_size_x + DC_pad_gap)

    # Ports
    c.add_port(
        name="e1",
        center=(0, DC_pad_size_x / 2),
        width=1,
        orientation=90,
        layer=layer_MT2,
        port_type="electrical"
    )
    c.add_port(
        name="e2",
        center=(DC_pad_size_x + DC_pad_gap, DC_pad_size_x / 2),
        width=1,
        orientation=90,
        layer=layer_MT2,
        port_type="electrical"
    )

    return c


@gf.cell
def fringeCapDOEcell(n_fingers=15, len_finger=150, w_finger=3, gap_finger=6, RFgPadw=70, RFPadw=60, RFpad_y_offset=0):
    # fid=open(ocdrDOE+'_log.txt','a')
    c = gf.Component('fringeCap')

    cap_width = n_fingers * gap_finger*2
    cap_center_x = cap_width / 2
    pad_buffer_x = gap_finger + w_finger + 1

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************

    # place RF pads
    RFgpad1 = c << gf.components.pad(size=[RFgPadw, RFgPadw], layer=MT2, port_inclusion=1, port_orientation=90)
    RFgpad1Open = c << gf.components.pad(size=[RFgPadw - 10, RFgPadw - 10], layer=PAD)
    RFgpad1.move((-pad_buffer_x, RFpad_y_offset))
    RFgpad1Open.move((-pad_buffer_x, RFpad_y_offset))

    RFgpad2 = c << gf.components.pad(size=[RFgPadw, RFgPadw], layer=MT2, port_inclusion=1, port_orientation=90)
    RFgpad2Open = c << gf.components.pad(size=[RFgPadw - 10, RFgPadw - 10], layer=PAD)
    RFgpad2.move((cap_width + pad_buffer_x, RFpad_y_offset))
    RFgpad2Open.move((cap_width + pad_buffer_x, RFpad_y_offset))

    RFpad = c << gf.components.pad(size=[RFPadw, RFPadw], layer=MT2, port_inclusion=1, port_orientation=90)
    RFpadOpen = c << gf.components.pad(size=[RFPadw - 10, RFPadw - 10], layer=PAD)
    RFpad.move((cap_center_x, RFpad_y_offset))
    RFpadOpen.move((cap_center_x, RFpad_y_offset))
    c.add_port(name="e1", center=(cap_center_x, RFpad_y_offset), width=0.5, orientation=0, layer=MT2, port_type="electrical")
    c.add_port(name="e2", center=(0 - pad_buffer_x, RFpad_y_offset), width=0.5, orientation=0, layer=MT2, port_type="electrical")
    c.add_port(name="e3", center=(cap_width + pad_buffer_x, RFpad_y_offset), width=0.5, orientation=0, layer=MT2, port_type="electrical")

    # place fringeCap
    #finger set 1
    elem = gf.components.rectangle(size=(w_finger, len_finger), layer=MT2)
    aref = c.add_ref(elem, columns=n_fingers, rows=1, spacing=(gap_finger*2, 0))
    aref.move((cap_center_x - n_fingers * gap_finger, 60))
    #finger set 2
    elem = gf.components.rectangle(size=(w_finger, len_finger), layer=MT2)
    aref = c.add_ref(elem, columns=n_fingers, rows=1, spacing=(gap_finger*2, 0))
    aref.move((cap_center_x - n_fingers * gap_finger + gap_finger, 62))
    #top and bottom electrode bars
    elem = gf.components.rectangle(size=(n_fingers * gap_finger*2, 4), layer=MT2)
    aref = c.add_ref(elem, columns=1, rows=2, spacing=(0, len_finger + 4+2))
    aref.move((cap_center_x - n_fingers * gap_finger, 60 - 4))

    c.add_port(name="e4", center=(cap_center_x, 62 - gap_finger - 16), width=0.5, orientation=0, layer=MT2, port_type="electrical") #lower left
    c.add_port(name="e5", center=(cap_center_x, 62 + len_finger + 5), width=0.5, orientation=0, layer=MT2, port_type="electrical") #upper left
    # *****************************************************************************************
    #	routing
    # *****************************************************************************************
    routes = [gf.routing.get_bundle_from_steps(
        c.ports["e1"], c.ports["e4"],
        width=40, layer=MT2,
        cross_section=gf.cross_section.metal1)[0]]
    routes.append(gf.routing.get_bundle_from_steps(
        c.ports["e2"], c.ports["e5"],
        width=10, layer=MT2,
        cross_section=gf.cross_section.metal1)[0])
    routes.append(gf.routing.get_bundle_from_steps(
        c.ports["e3"], c.ports["e5"],
        width=10, layer=MT2,
        cross_section=gf.cross_section.metal1)[0])
    for i in routes:
        c.add(i.references)
    # show die
    # c.show()
    # fid.close()
    return c


@gf.cell
def microscopeDOEcell(w, Padw=70, StripLength=1000):
    # fid=open(ocdrDOE+'_log.txt','a')

    c = gf.Component('microscope')

    # *****************************************************************************************
    #	component placement
    # *****************************************************************************************

    # place pads
    pad1 = c << gf.components.pad(size=[Padw, Padw + StripLength], layer=MT2, port_inclusion=1, port_orientation=90)
    pad1Open = c << gf.components.pad(size=[Padw - 10, Padw + StripLength - 10], layer=PAD)
    pad1.move((0, StripLength / 2))
    pad1Open.move((0, StripLength / 2))
    pad2 = c << gf.components.pad(size=[Padw, Padw], layer=MT2, port_inclusion=1, port_orientation=90)
    pad2Open = c << gf.components.pad(size=[Padw - 10, Padw - 10], layer=PAD)
    pad2.move((100, 0))
    pad2Open.move((100, 0))
    pad3 = c << gf.components.pad(size=[Padw, Padw + StripLength], layer=MT2, port_inclusion=1, port_orientation=90)
    pad3Open = c << gf.components.pad(size=[Padw - 10, Padw + StripLength - 10], layer=PAD)
    pad3.move((200, StripLength / 2))
    pad3Open.move((200, StripLength / 2))
    # add extensions
    pad1a = c << gf.components.pad(size=[(130 - w) / 2, Padw + StripLength - 100], layer=MT2, port_inclusion=1,
                                   port_orientation=90)
    pad1a.move((Padw / 2 + (130 - w) / 4, StripLength / 2 + 50))
    pad1aOpen = c << gf.components.pad(size=[(130 - w) / 2 + 10, Padw + StripLength - 100 - 10], layer=PAD,
                                       port_inclusion=1, port_orientation=90)
    pad1aOpen.move((Padw / 2 + (130 - w) / 4 - 10, StripLength / 2 + 50))
    pad3a = c << gf.components.pad(size=[(130 - w) / 2, Padw + StripLength - 100], layer=MT2, port_inclusion=1,
                                   port_orientation=90)
    pad3a.move((200 - (Padw / 2 + (130 - w) / 4), StripLength / 2 + 50))
    pad3aOpen = c << gf.components.pad(size=[(130 - w) / 2 + 10, Padw + StripLength - 100 - 10], layer=PAD,
                                       port_inclusion=1, port_orientation=90)
    pad3aOpen.move((200 - (Padw / 2 + (130 - w) / 4 - 10), StripLength / 2 + 50))
    # place M1 backplate
    pad2a = c << gf.components.pad(size=[300, StripLength], layer=MT1, port_inclusion=1, port_orientation=90)
    pad2a.move((100, StripLength / 2))
    # place vias
    c.add_polygon([(100 - 1.5, -1.5 + 10), (100 + 1.5, -1.5 + 10), (100 + 1.5, 1.5 + 10), (100 - 1.5, +1.5 + 10)],
                  layer=VIA2)
    c.add_polygon([(100 - 1.5 - 10, -1.5 + 10), (100 + 1.5 - 10, -1.5 + 10), (100 + 1.5 - 10, 1.5 + 10),
                   (100 - 1.5 - 10, +1.5 + 10)], layer=VIA2)
    c.add_polygon([(100 - 1.5 + 10, -1.5 + 10), (100 + 1.5 + 10, -1.5 + 10), (100 + 1.5 + 10, 1.5 + 10),
                   (100 - 1.5 + 10, +1.5 + 10)], layer=VIA2)

    # *****************************************************************************************
    #	routing
    # *****************************************************************************************

    # show die
    # c.show()
    # fid.close()
    return c


'''TEST STRUCTURES'''


@gf.cell
def GSG_test_straight(pad_center_gnd_width, pad_inner_gap_width,  # pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width,
                      test_length, pad_length, params:dict):
    ### PS signal traces == narrow, medium or wide
    PS_center_gnd_width = params["PS_center_gnd_width"]  # we should just call the params directly when needed..
    PS_inner_gap_width = params["PS_inner_gap_width"]
    PS_sig_width = params["PS_sig_width"]
    PS_outer_gap_width = params["PS_outer_gap_width"]
    PS_outer_gnd_width = params["PS_outer_gnd_width"]

    S2S_center_gnd_width = params["S2S_center_gnd_width"]
    S2S_inner_gap_width = params["S2S_inner_gap_width"]
    S2S_sig_width = params["S2S_sig_width"]
    S2S_outer_gap_width = params["S2S_outer_gap_width"]
    S2S_outer_gnd_width = params["S2S_outer_gnd_width"]
    S2S_length = params["S2S_length"]

    if pad_center_gnd_width is None:
        pad_center_gnd_width = params["pad_center_gnd_width"]
        pad_inner_gap_width = params["pad_inner_gap_width"]
        pad_sig_width = params["pad_sig_width"]
        pad_outer_gap_width = params["pad_outer_gap_width"]
        pad_outer_gnd_width = params["pad_outer_gnd_width"]
        pad_length = params["pad_length"]

    sc_length = params["sc_length"]

    spacing_x = params["spacing_x"]
    spacing_y = params["spacing_y"]
    DC_pad_size_x = params["DC_pad_size_x"]
    DC_pad_size_y = params["DC_pad_size_y"]
    pads_rec_gap = params["pads_rec_gap"]
    layer_HTR = params["layer_HTR"]
    layer_VIA2 = params["layer_VIA2"]
    trans_length = params["trans_length"]

    pad_sig_width = pad_center_gnd_width
    pad_outer_gap_width = pad_inner_gap_width
    pad_outer_gnd_width = pad_center_gnd_width

    c = gf.Component()

    # input contact pads
    offset_y = pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2
    straight, x1 = GSG_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, test_length, MT2)
    _ = c << straight
    _.move((0, offset_y))
    pad_in_PAD, x150 = GSG_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
    _ = c << pad_in_PAD
    _.move((5, offset_y))
    pad_out_PAD, x151 = GSG_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
    _ = c << pad_out_PAD
    _.move((test_length - pad_length + 5, offset_y))

    c.add_port(
        name="e_low",
        center=(pad_length + trans_length + S2S_length,
                -PS_center_gnd_width / 2 - PS_inner_gap_width / 2),
        width=1,
        orientation=0,
        layer=MT2,
        port_type="electrical",
    )

    c.add_port(
        name="e_up",
        center=(pad_length + trans_length + S2S_length,
                PS_center_gnd_width / 2 + PS_inner_gap_width / 2),
        width=1,
        orientation=0,
        layer=MT2,
        port_type="electrical",
    )

    return c

@gf.cell
def GSGSG_test_straight(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width,
                        pad_outer_gap_width, pad_outer_gnd_width, test_length, pad_length, params:dict):

    ### PS signal traces == narrow, medium or wide
    PS_center_gnd_width = params["PS_center_gnd_width"]  # we should just call the params directly when needed..
    PS_inner_gap_width = params["PS_inner_gap_width"]
    PS_sig_width = params["PS_sig_width"]
    PS_outer_gap_width = params["PS_outer_gap_width"]
    PS_outer_gnd_width = params["PS_outer_gnd_width"]

    S2S_center_gnd_width = params["S2S_center_gnd_width"]
    S2S_inner_gap_width = params["S2S_inner_gap_width"]
    S2S_sig_width = params["S2S_sig_width"]
    S2S_outer_gap_width = params["S2S_outer_gap_width"]
    S2S_outer_gnd_width = params["S2S_outer_gnd_width"]
    S2S_length = params["S2S_length"]
    if pad_center_gnd_width is None:
        pad_center_gnd_width = params["pad_center_gnd_width"]
        pad_inner_gap_width = params["pad_inner_gap_width"]
        pad_sig_width = params["pad_sig_width"]
        pad_outer_gap_width = params["pad_outer_gap_width"]
        pad_outer_gnd_width = params["pad_outer_gnd_width"]
        pad_length = params["pad_length"]

    sc_length = params["sc_length"]
    spacing_x = params["spacing_x"]
    spacing_y = params["spacing_y"]
    DC_pad_size_x = params["DC_pad_size_x"]
    DC_pad_size_y = params["DC_pad_size_y"]
    pads_rec_gap = params["pads_rec_gap"]
    layer_HTR = params["layer_HTR"]
    layer_VIA2 = params["layer_VIA2"]
    trans_length=params["trans_length"]

    c = gf.Component()

    # input contact pads
    straight, x1 = SGS_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, test_length, MT2)
    _ = c << straight
    pad_in_PAD, x150 = SGS_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
    _ = c << pad_in_PAD
    _.movex(5)
    pad_out_PAD, x151 = SGS_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
    _ = c << pad_out_PAD
    _.movex(test_length - pad_length + 5)

    c.add_port(
        name="e_low",
        center=(pad_length + trans_length + S2S_length,
                -PS_center_gnd_width / 2 - PS_inner_gap_width / 2),
        width=1,
        orientation=0,
        layer=MT2,
        port_type="electrical",
    )

    c.add_port(
        name="e_up",
        center=(pad_length + trans_length + S2S_length,
                PS_center_gnd_width / 2 + PS_inner_gap_width / 2),
        width=1,
        orientation=0,
        layer=MT2,
        port_type="electrical",
    )

    #GSGSG is missing appropriate ports?

    return c

def GSG_test_group_shared_gnd(GSGDOE_csv:str, num_pairs):
    '''

    going up
    etc.
    SG
    SG
    then SG
    define G
    '''

    def G_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, pad_length, layer_MT):
        p = gf.path.straight(length=pad_length, npoints=2)
        s0 = gf.Section(width=pad_center_gnd_width, offset=0, layer=layer_MT, name="center gnd", port_names=("e1", "e2"), port_types=("electrical", "electrical"))
        x1 = gf.CrossSection(sections=(s0,))
        generate = gf.path.extrude(p, x1)
        return generate, x1


    def SG_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, pad_length, layer_MT):
        p = gf.path.straight(length=pad_length, npoints=2)
        off_s1 = 0#-(pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width / 2)
        s1 = gf.Section(width=pad_sig_width, offset=off_s1, layer=layer_MT, name="sig1")
        off_s2 = -(pad_sig_width/2 + pad_outer_gap_width + pad_outer_gnd_width / 2)
        #off_s2 = -(pad_center_gnd_width / 2 + pad_inner_gap_width + pad_sig_width + pad_outer_gap_width + pad_outer_gnd_width / 2)
        s2 = gf.Section(width=pad_outer_gnd_width, offset=off_s2, layer=layer_MT, name="outer gnd1", port_names=("e1", "e2"), port_types=("electrical", "electrical"))
        x1 = gf.CrossSection(sections=(s1, s2))
        generate = gf.path.extrude(p, x1)
        return generate, x1

    @gf.cell
    def G_test_straight(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, #same as SG_test_straight but with G_piece(..) #create G straight with pads
                        pad_outer_gap_width, pad_outer_gnd_width, test_length, pad_length):
        c = gf.Component("G_test_straight")
        straight, x1 = G_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, test_length, MT2)
        _ = c << straight
        pad_in_PAD, x150 = G_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
        _ = c << pad_in_PAD
        _.movex(5)
        pad_out_PAD, x151 = G_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
        _ = c << pad_out_PAD
        _.movex(test_length - pad_length + 5)

        c.add_port(
            name="e_low",
            center=(pad_length),
            width=1,
            orientation=0,
            layer=MT2,
            port_type="electrical",
        )
        c.add_port(
            name="e_up",
            center=(pad_length),
            # center=(pad_length + S2S_length + trans_length,
            #         PS_center_gnd_width / 2 + PS_inner_gap_width / 2),
            width=1,
            orientation=0,
            layer=MT2,
            port_type="electrical",
        )
        return c

    @gf.cell
    def SG_test_straight(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, #create SG straights with pads
                        pad_outer_gap_width, pad_outer_gnd_width, test_length, pad_length):
        c = gf.Component("SG_test_straight")
        straight, x1 = SG_piece(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width, pad_outer_gap_width, pad_outer_gnd_width, test_length, MT2)
        _ = c << straight
        pad_in_PAD, x150 = SG_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
        _ = c << pad_in_PAD
        _.movex(5)
        pad_out_PAD, x151 = SG_piece(pad_center_gnd_width - 10, pad_inner_gap_width + 10, pad_sig_width - 10, pad_outer_gap_width + 10, pad_outer_gnd_width - 10, pad_length - 10, PAD)
        _ = c << pad_out_PAD
        _.movex(test_length - pad_length + 5)

        # c.add_port(
        #     name="e_low",
        #     center=(pad_length + trans_length + S2S_length,
        #             -PS_center_gnd_width / 2 - PS_inner_gap_width / 2),
        #     width=1,
        #     orientation=0,
        #     layer=MT2,
        #     port_type="electrical",
        # )
        # c.add_port(
        #     name="e_up",
        #     center=(pad_length + trans_length + S2S_length,
        #             PS_center_gnd_width / 2 + PS_inner_gap_width / 2),
        #     width=1,
        #     orientation=0,
        #     layer=MT2,
        #     port_type="electrical",
        # )
        return c



    c = gf.Component(GSGDOE_csv)

    doe = pd.read_csv(GSGDOE_csv)
    sw, device = [], []
    pos = 0

    #first G piece at bottom
    i = 0 #copy G parameters from first SG
    pad_center_gnd_width = doe['pad_center_gnd_width(um)'][i]
    pad_inner_gap_width = doe['pad_inner_gap_width(um)'][i]
    pad_sig_width = doe['pad_sig_width(um)'][i]
    pad_outer_gap_width = doe['pad_outer_gap_width(um)'][i]
    pad_outer_gnd_width = doe['pad_outer_gnd_width(um)'][i]
    pad_length = doe['pad_length(um)'][i]
    test_length = doe['test_length(um)'][i]
    _ = c << G_test_straight(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width,
                                       pad_outer_gap_width, pad_outer_gnd_width, test_length, pad_length)
    _.move((0, pos))
    pos = pos + _.ymax
    print(pos)
    ###
    #further SG pieces, stacking up
    for i, j in doe.iterrows():
        # skip cells
        if i < num_pairs:
            pad_center_gnd_width = doe['pad_center_gnd_width(um)'][i]
            pad_inner_gap_width = doe['pad_inner_gap_width(um)'][i]
            pad_sig_width = doe['pad_sig_width(um)'][i]
            pad_outer_gap_width = doe['pad_outer_gap_width(um)'][i]
            pad_outer_gnd_width = doe['pad_outer_gnd_width(um)'][i]
            pad_length = doe['pad_length(um)'][i]
            test_length = doe['test_length(um)'][i]

            device.append(c << SG_test_straight(pad_center_gnd_width, pad_inner_gap_width, pad_sig_width,
                                               pad_outer_gap_width, pad_outer_gnd_width, test_length, pad_length))

            device[i].move((0, pos + pad_center_gnd_width/2 + pad_inner_gap_width))
            pos = device[i].ymax # (pad_sig_width / 2 + pad_outer_gap_width + pad_outer_gnd_width / 2)
            print(device[i].ymax )

    ###
    return c


@gf.cell
def GSGSG_resistor(termination, params:dict):
    #straight version

    layer_MT2 = params["layer_MT2"]
    layer_PAD = params["layer_PAD"]
    layer_HTR = params["layer_HTR"]
    layer_VIA2 = params["layer_VIA2"]

    c = gf.Component()

    # Terminating heater
    htr_length = 50
    htr_width_35 = 8.562
    htr_width_50 = 5.995
    htr_width_65 = 4.613
    htr_connect_length = 20

    # Terminating pads
    pad_t_center_gnd_width = 100
    pad_t_inner_gap_width = 20
    pad_t_sig_width = 80
    pad_t_outer_gap_width = 20
    pad_t_outer_gnd_width = 86
    pad_t_length = 100

    ### termination resistance (Ohm)
    if termination == 35:
        pad_t_center_gnd_width = 120
        pad_t_inner_gap_width = 7.5
        pad_t_sig_width = 100
        pad_t_outer_gap_width = 7.5
        pad_t_outer_gnd_width = 200
        htr_width_local = htr_width_35

    elif termination == 50:
        pad_t_center_gnd_width = 100
        pad_t_inner_gap_width = 20
        pad_t_sig_width = 80
        pad_t_outer_gap_width = 20
        pad_t_outer_gnd_width = 86
        htr_width_local = htr_width_50

    elif termination == 65:
        pad_t_center_gnd_width = 60
        pad_t_inner_gap_width = 45
        pad_t_sig_width = 70
        pad_t_outer_gap_width = 45
        pad_t_outer_gnd_width = 70
        htr_width_local = htr_width_65

    else:
        print("check termination resistance")

    ### setting up components
    # output contact pads
    pad_t_out, x5 = SGS_piece(pad_t_center_gnd_width, pad_t_inner_gap_width, pad_t_sig_width, pad_t_outer_gap_width, pad_t_outer_gnd_width, pad_t_length, layer_MT2)
    _ = c << pad_t_out
    _.movex()
    pad_t_out_PAD, x150 = SGS_piece(pad_t_center_gnd_width - 10, pad_t_inner_gap_width + 10, pad_t_sig_width - 10, pad_t_outer_gap_width + 10, pad_t_outer_gnd_width - 10, pad_t_length - 10, layer_PAD)
    _ = c << pad_t_out_PAD
    _.movex(5)

    # Heater connection pads
    htr_connect, x1511 = SGS_piece(pad_t_center_gnd_width - htr_length + pad_t_inner_gap_width, htr_length, pad_t_sig_width - htr_length + (+pad_t_outer_gap_width + pad_t_outer_gap_width) / 2, htr_length,
                                     pad_t_outer_gnd_width - htr_length + pad_t_outer_gap_width / 2, htr_connect_length, layer_HTR)
    _ = c << htr_connect
    _.movex(pad_t_length / 2 - htr_connect_length / 2)

    # Heaters
    heaters, x1512 = SGS_piece(0, (pad_t_center_gnd_width - htr_length + pad_t_inner_gap_width) / 2, htr_length, pad_t_sig_width - htr_length + (+pad_t_outer_gap_width + pad_t_outer_gap_width) / 2, htr_length,
                                 htr_width_local, layer_HTR)
    _ = c << heaters
    _.movex(pad_t_length / 2 - htr_width_local / 2)

    # Via2 between MT2 and HTR
    via2, x120 = SGS_piece(pad_t_center_gnd_width - htr_length + pad_t_inner_gap_width - 3, htr_length + 3, pad_t_sig_width - htr_length + (+pad_t_outer_gap_width + pad_t_outer_gap_width) / 2 - 3, htr_length + 3,
                             pad_t_outer_gnd_width - htr_length + pad_t_outer_gap_width / 2 - 3, htr_connect_length - 3, layer_VIA2)
    _ = c << via2
    _.movex(pad_t_length / 2 - htr_connect_length / 2 + 1.5)

    return c


@gf.cell
def GSG_DOE(params:dict):
    spacing_x, spacing_y = 200, 200
    ### DOE parameters setting
    DOE_full = []
    DOE_row = []
    count = 0
    for j in range(200, 800, 200):
        DOE_row = []
        if count % 2 == 0:
            for i in range(200, 1400, 200):
                params={**params, "PS_length": i, "trans_length": j}
                DOE_row.append(GSG_MT2(params))
            #DOE = [GSG_MZM(PS_length=i, trans_length=j) for i in range(200, 1400, 200)]
        else:
            for i in range(1200, 0, -200):
                params = {**params, "PS_length": i, "trans_length": j}
                DOE_row.append(GSG_MT2(params))
            #DOE = [GSG_MZM(PS_length=i, trans_length=j) for i in range(1200, 0, -200)]
        m = gf.grid(DOE_row, shape=(len(DOE_row), 1), rotation=0, spacing=(spacing_x, spacing_y))
        DOE_full.append(DOE_row)
        count += 1

    ### generate array
    m = gf.grid(DOE_full, shape=(len(DOE_full), len(DOE_row)), rotation=270, spacing=(spacing_x, spacing_y - 100 * (len(DOE_row) - 1)))
    m.name = "GSG_DOE"
    return m

@gf.cell
def GSGSG_DOE(params:dict):
    spacing_x, spacing_y = 200, 200
    ### DOE parameters setting
    DOE_full = []
    DOE_row = []
    count = 0
    for j in range(200, 800, 200):
        DOE_row = []
        if count % 2 == 0:
            for i in range(200, 1400, 200):
                DOE_row.append(GSGSG_MT2_symmetric(PS_length=i, trans_length=j, taper_type=params["taper_type"], sig_trace=params["sig_trace"], params=params))
            #DOE = [GSG_MZM(PS_length=i, trans_length=j) for i in range(200, 1400, 200)]
        else:
            for i in range(1200, 0, -200):
                DOE_row.append(GSGSG_MT2_symmetric(PS_length=i, trans_length=j, taper_type=params["taper_type"], sig_trace=params["sig_trace"], params=params))
            #DOE = [GSG_MZM(PS_length=i, trans_length=j) for i in range(1200, 0, -200)]
        m = gf.grid(DOE_row, shape=(len(DOE_row), 1), rotation=0, spacing=(spacing_x, spacing_y))
        DOE_full.append(DOE_row)
        count += 1

    ### generate array
    m = gf.grid(DOE_full, shape=(len(DOE_full), len(DOE_row)), rotation=270, spacing=(spacing_x, spacing_y - 100 * (len(DOE_row) - 1)))
    m.name = "GSGSG_DOE"
    return m
