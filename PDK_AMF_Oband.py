import gdsfactory as gf
from layer_info import *
import math


c = gf.Component("PDK_test")


MMI_2X2_O = gf.import_gds("AMF_SOI_Oband_pdk_4p5_bb_20240405.gds", "AMF_SOI_2X2MMI_Oband_v3p1", with_metadata=True)
MMI_2X2_O = gf.add_ports.add_ports_from_markers_inside(MMI_2X2_O, pin_layer=(1010,0), port_layer=RIB)

EC_O = gf.import_gds("AMF_SOI_Oband_pdk_4p5_bb_20240405.gds", "AMF_SOI_EdgeCoupler_Oband_v3p1", with_metadata=True)
EC_O = gf.add_ports.add_ports_from_markers_inside(EC_O, pin_layer=(1010,0), port_layer=RIB)

GC_O = gf.import_gds("AMF_SOI_Oband_pdk_4p5_bb_20240405.gds", "AMF_SOI_GC1D_Oband_v3p0", with_metadata=True)
GC_O = gf.add_ports.add_ports_from_markers_inside(GC_O, pin_layer=(1010,0), port_layer=RIB)



TO_PS_O = gf.import_gds("AMF_SOI_Oband_pdk_4p5_bb_20240405.gds", "AMF_SOI_TOPhaseShifter1_Oband_v4p5", with_metadata=True)
TO_PS_O = gf.add_ports.add_ports_from_markers_inside(TO_PS_O, pin_layer=(1010,0), port_layer=RIB)
TO_PS_O.add_port(name="e1", center=(49, 50), width=20, orientation=90, layer=MT2, port_type = "electrical")
TO_PS_O.add_port(name="e2", center=(251, 50), width=20, orientation=90, layer=MT2, port_type = "electrical")


@gf.cell 
def EC_array_O(N, pitch):
    c = gf.Component("EC_array")
    for i in range(N):
        ref_EC = c << EC_O
        ref_EC.rotate(-90).move((i*pitch, 0))
        j = i + 1
        c.add_port(f"o{j}", port=ref_EC["o1"])  
#    c.pprint_ports()
    return c
    

@gf.cell 
def GC_array_v1_O(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC_O
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    length=2
    width=0.41
    _1 = c << gf.components.straight(length=length, width=width, layer=RIB)
    _1.connect('o1', c.ports["o1"])
    _2 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, width=width, layer=RIB)
    _2.connect('o2', _1.ports["o2"])
   
    _3 = c << gf.components.straight(length=length, width=width, layer=RIB)
    _3.connect('o1', c.ports["o2"])
    _4 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, width=width, layer=RIB)
    _4.connect('o1', _3.ports["o2"])   

    routes = gf.routing.get_bundle( _4.ports['o2'], _2.ports['o1'], layer = RIB)
    c.add(routes[0].references)

    return c
 

@gf.cell 
def GC_array_v2_O(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC_O
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    length=2
    width=0.41
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

