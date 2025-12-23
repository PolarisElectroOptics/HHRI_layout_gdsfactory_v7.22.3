import gdsfactory as gf
import math
from layer_info import *
import library_sipho_unified

c = gf.Component("PDK_test")


MMI_2X2 = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_2X2MMI_Cband_v3p0", with_metadata=True)
MMI_2X2 = gf.add_ports.add_ports_from_markers_inside(MMI_2X2, pin_layer=(1010,0), port_layer=RIB)

MMI_1X2 = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_1X2MMI_Cband_v3p0", with_metadata=True)
MMI_1X2 = gf.add_ports.add_ports_from_markers_inside(MMI_1X2, pin_layer=(1010,0), port_layer=RIB)


EC = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_EdgeCoupler_Cband_v3p0", with_metadata=True)
EC = gf.add_ports.add_ports_from_markers_inside(EC, pin_layer=(1010,0), port_layer=RIB)

GC = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_GC1D_Cband_v3p0", with_metadata=True)
GC = gf.add_ports.add_ports_from_markers_inside(GC, pin_layer=(1010,0), port_layer=RIB)



TO_PS = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_TOPhaseShifter1_Cband_v4p0", with_metadata=True)
TO_PS = gf.add_ports.add_ports_from_markers_inside(TO_PS, pin_layer=(1010,0), port_layer=RIB)
TO_PS.add_port(name="e1", center=(49, 50), width=20, orientation=90, layer=MT2, port_type = "electrical")
TO_PS.add_port(name="e2", center=(251, 50), width=20, orientation=90, layer=MT2, port_type = "electrical")


CRO = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_Crossing_Cband_v3p0", with_metadata=True)
CRO = gf.add_ports.add_ports_from_markers_inside(CRO, pin_layer=(1010,0), port_layer=RIB)

PBS = gf.import_gds("AMF_SOI_Cband_pdk_4p5_bb_20240405.gds", "AMF_SOI_PBS_Cband_v3p0", with_metadata=True)
PBS = gf.add_ports.add_ports_from_markers_inside(PBS, pin_layer=(1010,0), port_layer=RIB)

@gf.cell 
def EC_array(N, pitch):
    c = gf.Component("EC_array")
    for i in range(N):
        ref_EC = c << EC
        ref_EC.rotate(-90).move((i*pitch, 0))
        j = i + 1
        c.add_port(f"o{j}", port=ref_EC["o1"])  
#    c.pprint_ports()
    return c


@gf.cell 
def EC_array_gap(N, pitch, n):
    c = gf.Component("EC_array_gap")
    for i in range(math.floor(N/n)):
        for _ in range(n):
            ref_EC = c << EC
            ref_EC.rotate(-90).move(( (_*pitch) +(n+1)*i*pitch , 0))
            j = _ + 1 + (i*n)
            c.add_port(f"o{j}", port=ref_EC["o1"])  
    #    c.pprint_ports()
    return c
 

@gf.cell 
def GC_single():
    c = gf.Component("GC_single")
    ref_GC = c << GC
    
    return c


@gf.cell 
def GC_array_v1(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    length=2
    width=0.5
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
def GC_array_v2(N, pitch):
    c = gf.Component("GC_array")
    for i in range(N+2):
        ref_GC = c << GC
        ref_GC.rotate(0).move((0, i*pitch))
        j = i + 1
        c.add_port(f"o{j}", port=ref_GC["o1"])  
#    c.pprint_ports()

    length=2
    width=0.5
    _1 = c << gf.components.straight(length=length, cross_section=library_sipho_unified.rib_Oband)
    _1.connect('o1', c.ports[f"o{N+1}"] )
    _2 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, cross_section=library_sipho_unified.rib_Oband)
    _2.connect('o2', _1.ports["o2"] )
   
    _3 = c << gf.components.straight(length=length, cross_section=library_sipho_unified.rib_Oband)
    _3.connect('o1', c.ports[f"o{N+2}"] )
    _4 = c << gf.components.bend_euler(radius=5, angle=90, npoints=40, cross_section=library_sipho_unified.rib_Oband)
    _4.connect('o1', _3.ports["o2"] )   

    routes = gf.routing.get_bundle( _4.ports['o2'], _2.ports['o1'], cross_section=library_sipho_unified.rib_Oband)
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
        ### labeling
        label_a = c.add_label(text="GC_" + str(N+2-i), position=(GC.ysize/2, i*pitch), magnification=30, rotation=180, layer=(1011, 0), x_reflection=True)
        #label_a.mirror((1, 0)).rotate(0).move((-10, i*pitch))
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



# a = c << EC_array(4, 250)
# ref_TO = c << TO_PS
# c.show(show_ports=True)
# ref_TO.pprint_ports()
#print(MMI_2X2.ports)

# _ = c << GC_array_v2(3, 127)
# c.show()
# _.pprint_ports()

# _ = c << CRO
# ref_CRO = c << CRO
# c.show(show_ports=True)
# _.pprint_ports()

# _ = c << logo_Altair()
# c.show()