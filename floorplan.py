# this file contains parameters of the floorplan (die sizes, dicing lane, etc.)
# rev 0.1 - 28Dec2024


### Frame parameters
Die6_frm_w = 6000   
Die6_frm_len = 11100 
Die7_frm_w = 6000   
Die7_frm_len = 11100 
wscl1_frm_len = 3800
wscl1_frm_w = 6000
wscl2_frm_len = 3800
wscl2_frm_w = 6000
trench_w = 100

### Edge coupler pitch
EC_pitch = 127

### MZM placement
mzm_gap_edge = 5
mzm_pitch_x = 850
mzm_place_x = Die6_frm_w/2 - mzm_pitch_x*1.5+120
mzm_place_y = trench_w + mzm_gap_edge