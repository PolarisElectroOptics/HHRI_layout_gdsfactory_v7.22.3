### layer info - SilTerra EMO1 Rev9 12/22/2025

WG_HM = (275, 0)
WG_Strip = (101, 251)
WG_LowRib = (100, 90)

Impl_Window = (127, 60)
N_Impl = (257, 0)
NP_Impl = (257, 0)
NPP_Impl = (261, 0)

Salicide = (128, 60)
Contact_Si = (268, 0)

Heater = (29, 30)

UTM = (173, 0)  # thickness = 0.7 um
UTV = (172, 0)  # via between UTM & UTM2, thickness = 1.1 um
UTM2 = (197, 0)  # thickness = 2.1 um

## passivation = 2 um oxide and then 0.5 um nitride
Pad_Optical = (100, 160)  # remove passivation nitride
Pad_Electrical = (100, 170)  # then remove passivation oxide
PadAl = (145, 0)

Oxide_Open = (171, 60)

Oxide_Facet = (90, 0)

DeepTrench = (90, 0)

DM_Excl = (23, 0)

Exclusion = (57, 0)






SLOT_ETCH = (30, 2)
<<<<<<< HEAD
<<<<<<< Updated upstream
RIB = (275, 0)      #WG_HM
RIB_ETCH = (101, 251) #WG_Strip
=======
RIB = WG_HM      #FETCH_COR
RIB_ETCH = WG_Strip #FETCH_CLD
>>>>>>> Stashed changes
=======
RIB = WG_HM     #WG_HM
RIB_ETCH = WG_Strip #WG_Strip
>>>>>>> f6d59fb39a039d5cddd19a3586e0db02ffae5e76

SLAB = WG_LowRib     #WG_LowRib
SLAB_COR=(33, 1)   #METCH_COR   #not used SilTerra

SWG_DUMMY_BLOCK = (92, 0) #Replace w NOFILL -

GRAT = (11, 121)   #Not used in Crealights.


SB = (36, 0)        #SA (silicide)
NIM = (257, 0)     #NLD              # Si implant N
PIM = (24, 0)      #PLD      # Si implant P

IND = (257, 0)      #NW   # Si intermediate N
IPD = (22, 0)      #PW  # Si intermediate P

NCONT = (261, 0)     #NP             # Si implant N++
PCONT = (28, 0)     #PP             # Si implant P++

CONTACT = (268, 0)   #CA (contact)
VIA1 = UTV      #via1
VIA2 = (27, 0)      #via2 #NOT USED IN SilTerra

MT1 = UTM        #metal1
MT1_SLOT = (11, 20)
MT1_DUMMY_BLOCK = (11, 11)
MT2 = UTM2     #metal2
MT2_SLOT = (12, 20)
MT2_DUMMY_BLOCK = (12, 11)

MT3 = (28, 0)       #metal3 #not used in crealights
MT3_SLOT = (28, 20)
MT3_DUMMY_BLOCK = (28, 30)

PAD = (60, 0)        #PASS1
#CUPAD = (118, 132)  #not used

OXOP = Oxide_Open     #OPEN
NOP = (37, 2)      #nitride open   #NFE_CLD

HTR = (19, 0)       #heater

DTR = (160, 0)      #not used

DETCH = (64, 0)     #DETCH #deep etch for edge couplers
DAM = (333, 0)      #NEED TO UPDATE

#WGCNCT = (118, 120) #rib/slab transition region

ARTIFACT = (90, 0) #LOGO, need to modify

#DATAEXTENT = (118, 134)