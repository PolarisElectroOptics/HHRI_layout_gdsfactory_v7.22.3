import gdsfactory as gf
import math
from layer_info import *
import os
from numpy import arctan2, degrees, isclose
from gdsfactory.port import Port, read_port_markers, sort_ports_clockwise
#from library_sipho_unified import *
from gdsfactory.typings import LayerSpec
from gdsfactory.component import Component

port_electrical = (11,1) #temporary

@gf.cell
def add_ports_from_Crealight(
    component: Component,
    pin_layer_optical: LayerSpec = "PORT",
    port_layer_optical: LayerSpec | None = None,
    pin_layer_electrical: LayerSpec = "PORTE",
    port_layer_electrical: LayerSpec | None = None,
    label_tag = ['wg'],
    port_type = ['electrical', 'optical'],
) -> Component:
    """Add ports from SiEPIC-type cells, where the pins are defined as paths.

    Looks for label, path pairs.

    Args:
        component: component.
        pin_layer_optical: layer for optical pins.
        port_layer_optical: layer for optical ports.
        pin_layer_electrical: layer for electrical pins.
        port_layer_electrical: layer for electrical ports.
        label_tag  : list containing the label for ports 
        port_type : list containing type of ports
    """
    pin_layers = {"optical": pin_layer_optical, "electrical": pin_layer_electrical}

    import gdsfactory as gf

    pin_layer_optical = gf.get_layer(pin_layer_optical)
    port_layer_optical = gf.get_layer(port_layer_optical)
    pin_layer_electrical = gf.get_layer(pin_layer_electrical)
    port_layer_electrical = gf.get_layer(port_layer_electrical)

    c = component
    labels = c.get_labels()
    
    # for it in labels:
    #     print(str(it.text))
    
    lab_n = 0

    for path in c.paths:
        p1, p2 = path.spine()

        path_layers = list(zip(path.layers, path.datatypes))

        # Find the center of the path
        center = (p1 + p2) / 2

        
        if  port_type[lab_n] == 'optical':
            port_type = "optical"
            port_layer = port_layer_optical 
        elif port_type[lab_n] == 'electrical':
           port_type = "electrical"
           port_layer = port_layer_electrical

        port_name = label_tag[lab_n]

        

        angle = round(degrees(arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 360)

        port = Port(
            name=port_name,
            center=center,
            width=20,#path.widths()[0][0],
            orientation=angle,
            layer=port_layer or pin_layers[port_type],
            port_type=port_type,
        )
        c.add_port(port)
        
        lab_n = lab_n + 1
    return c







direct_ = os.getcwd()
GC_TE = gf.import_gds(direct_ + "/black_boxs/grating_couplers/Fixed_GC_TE_1310.gds", with_metadata=True)
GC_TE = add_ports_from_Crealight(component = GC_TE, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1'],port_type = ['optical'])


GC_TM = gf.import_gds(direct_ + "/black_boxs/grating_couplers/Fixed_GC_TM_1310.gds", with_metadata=True)
GC_TM =  add_ports_from_Crealight(component = GC_TM, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1'],port_type = ['optical'])


GC_TE_SIN = gf.import_gds(direct_ + "/black_boxs/grating_couplers/Fixed_GC_TE_1310_SIN.gds", with_metadata=True)
GC_TE_SIN = add_ports_from_Crealight(component = GC_TE_SIN, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1'],port_type = ['optical'])


GC_2D_1310 = gf.import_gds(direct_ + "/black_boxs/grating_couplers/Fixed_GC_2D_1310.gds", with_metadata=True)
GC_2D_1310 = add_ports_from_Crealight(component = GC_2D_1310, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o_1','o_2'],port_type = ['optical','optical'])

EC_ST_FA0_1310_300_Nopolish_BBOX = gf.import_gds(direct_ + "/black_boxs/edge_couplers/EC_ST_FA0_1310_300_Nopolish_BBOX.gds", with_metadata=True)
EC = add_ports_from_Crealight(component =EC_ST_FA0_1310_300_Nopolish_BBOX, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1'],port_type = ['optical'])

EC_ST_FA8_1310_300_Nopolish_BBOX= gf.import_gds(direct_ + "/black_boxs/edge_couplers/EC_ST_FA8_1310_300_Nopolish_BBOX.gds", with_metadata=True)
EC_ST_FA8_1310_300_Nopolish_BBOX = add_ports_from_Crealight(component =EC_ST_FA8_1310_300_Nopolish_BBOX, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1'], port_type = ['optical'])

EC_ST_NFE_FA8_1310_300_Nopolish_BBOX= gf.import_gds(direct_ + "/black_boxs/edge_couplers/EC_ST_NFE_FA8_1310_300_Nopolish_BBOX.gds", with_metadata=True)
EC_ST_NFE_FA8_1310_300_Nopolish_BBOX = add_ports_from_Crealight(component =EC_ST_NFE_FA8_1310_300_Nopolish_BBOX, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1'],port_type = ['optical'])



Fixed_SiN_Si_TE_1310= gf.import_gds(direct_ + "/black_boxs/transitions/Fixed_SiN_Si_TE_1310_20250227.gds", with_metadata=True)
Fixed_SiN_Si_TE_1310 =add_ports_from_Crealight(component =Fixed_SiN_Si_TE_1310, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = [ 'o2', 'o1'], port_type = ['optical','optical'])


heater_unit_v1= gf.import_gds(direct_ + "/black_boxs/heaters/heater_unit_v1.gds", with_metadata=True)
TO_PS =add_ports_from_Crealight(component =heater_unit_v1, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = [ 'o1', 'o2', 'e1','e2', 'e3'],port_type = ['optical','optical','electrical','electrical','electrical' ])




MPD= gf.import_gds(direct_ + "/black_boxs/monitor PD/MPD.gds", with_metadata=True)
MPD =add_ports_from_Crealight(component =MPD, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = [ 'o1', 'e1', 'e2'],port_type = ['optical','electrical','electrical' ])

M1X2 = gf.import_gds(direct_ + "/black_boxs/MMIs/Fixed_M1X2_TE_1310.gds", with_metadata=True) 
MMI_1X2 = add_ports_from_Crealight(component = M1X2, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o1', 'o2','o3'],port_type = ['optical','optical','optical'])



M2X2 = gf.import_gds(direct_ + "/black_boxs/MMIs/Fixed_M2X2_TE_1310.gds", with_metadata=True) 
MMI_2X2 = add_ports_from_Crealight(component = M2X2, pin_layer_optical = (95,0), port_layer_optical= RIB, label_tag = ['o2', 'o1','o3','o4'],port_type = ['optical','optical','optical','optical'])

#MMI_2X2.show()


"""
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
    _2 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, cross_section = rib_Oband)
    _2.connect('o2', _1.ports["o2"])
   
    _3 = c << gf.components.straight(length=length,cross_section = rib_Oband)
    _3.connect('o1', c.ports["o2"])
    _4 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, cross_section = rib_Oband)
    _4.connect('o1', _3.ports["o2"])   

    routes = gf.routing.get_bundle( _4.ports['o2'], _2.ports['o1'],cross_section = rib_Oband)
    c.add(routes[0].references)

    return c

    



@gf.cell 
def GC_array_v2(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC_TE
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    length=2
    width=0.5
    _1 = c << gf.components.straight(length=length, width=width, layer=RIB)
    _1.connect('o1', c.ports[f"o{N+1}"] )
    _2 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, width=width, layer=RIB)
    _2.connect('o2', _1.ports["o2"] )
   
    _3 = c << gf.components.straight(length=length, width=width, layer=RIB)
    _3.connect('o1', c.ports[f"o{N+2}"] )
    _4 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, width=width, layer=RIB)
    _4.connect('o1', _3.ports["o2"] )   

    routes = gf.routing.get_bundle( _4.ports['o2'], _2.ports['o1'], layer = RIB)
    c.add(routes[0].references)

    return c   


@gf.cell 
def GC_array_v3(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    return c


@gf.cell 
def logo_Altair():
    c = gf.Component()
    
    _ = c << gf.import_gds("DocCell_20250115_flattened.gds", "doc", with_metadata=True)
    
    return c



@gf.cell 
def Dektak(length, width, block_len, pitch):
    c = gf.Component()
    
    op = c << gf.components.straight(length=length, width=width, layer=OXOP)
    
    N = math.floor(length/pitch/2/2)

    rib = gf.components.straight(length=block_len, width=width-10, layer=RIB)
    ribs = c << gf.components.array(component=rib, spacing=(pitch*2, 0), columns=N, rows=1)
    
    slab = gf.components.straight(length=block_len, width=width-10, layer=SLAB)
    slabs = c << gf.components.array(component=slab, spacing=(pitch*2, 0), columns=N, rows=1)
    slabs.move(( pitch, 0 ))
    
    ribs2 = c << gf.components.array(component=rib, spacing=(pitch*2, 0), columns=N, rows=1)
    ribs2.move(( length/2, 0 ))


    slabs2 = c << gf.components.array(component=slab, spacing=(pitch*2, 0), columns=N, rows=1)
    slabs2.move(( length/2 +block_len -0.1, 0 ))    
    
    return c

"""

# c = gf.Component("PDK_test")

# # a = c << EC_array(4, 250)
# # ref_TO = c << TO_PS
# # c.show(show_ports=True)
# # ref_TO.pprint_ports()
# #print(MMI_2X2.ports)

# _ = c << GC_array_v1(3, 127)
# #c.show()
# _.pprint_ports()

# d = merge_clad(c)

# d.show()


# _ = c << CRO
# ref_CRO = c << CRO
# c.show(show_ports=True)
# _.pprint_ports()

# _ = c << logo_Altair()
# c.show()