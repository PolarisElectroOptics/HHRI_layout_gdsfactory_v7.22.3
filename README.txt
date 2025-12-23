last updated: 12/1/2025

python: 3.10
GDSFactory: 7.22.3

This library is used to generate all of our GDS designs for Polaris.



We use snake case, and global variables and abbreviations will be in all caps. 
(RIB, SLAB, MT1, etc. ; MZM, GSG, GSGSG, S2S, PS, DOE)
S2S stands for slab to slot, it is the mode converter
PS stands for phase shifter
GSG, GSGSG are the electrode structures
MZM references the entire Mach-Zender Modulator, including all photonic and electronic parts



The structure of the library is as follows: 
layout_<Tapeout_name> - top level layout for entire tapeout, contains multiple dies

layout_die_<die_name> - die layout
layer_info  		  - contains GDS layer definitions for current fab
PDK_<pdk_name> 		  - contains black box PDK components from fab
pars_<parameter_file_name>.csv - parameter file for DOE of components
<klayout_layer_defintitions>.lyp - Klayout layer definition file for current fab

library_electrode_unified 	- library of functions that generate electrical components. Functions lower in the file may depend on earlier ones. 
library_electrode_params 	- electrical parameters to be called in conjunction with library_electrode_unified components
library_sipho_unified 		- library of functions that generate photonic components. Functions lower in the file may depend on earlier ones. 
library_sipo_params 			- photonic parameters to be called in conjunction with library_sipho_unified components

an example component hierarchy would look like this:
layout_Altair
	layout_die_DOE1
		MZM_GSGSG
			PS_connected
				S2S
				PS_SlotWG
		ocdr_DOE
		fringeCapDoecell
		routing functions
		logo_Altair





Documentation - this library uses doctrings under each function. 
Docstring should include example of how to use this component (calling the component function with relevant parameter library) along with basic description of component and key/required paramters. 
This README should be updated regularly as well. 



Gitub structure: 
In-house-library_v_7.22.3
	main - original tapeout code for Alair
	final_integration_Altair - Altair tapeout with unified library
	new-development - most up-to-date stable Altair branch
	molex-tower-PH18MD - Molex tapeout with Towerjazz. library components is on-par with new-development, with changes made for Tower process and cheesing function added
	Crealights-Tapeout-December-main - Crealights tapeout for December 2025. Based on molex-tower-PH18MD and updated for Crealights process.
	*GDSFactory9 branches - conversion to GDSFactory9, still WIP and not fully working.
	
Vega_layout_gdsfactory_v7.22.3 - Vega tapeout, originally planned with GF around Sept. 2025. Cancelled.
Mirach_layout_gdsfactory_v7.22.3 - Mirach tapeout, planned for Dec 2025 with Crealights. Planning to migrate Crealights-Tapeout-December-main to here after Shafiqul merges his branch, should be soon. (around 12.2.2025)

Layout_gdsfactory - repository with code examples for GDSFactory 7.22.3
	
	
	
