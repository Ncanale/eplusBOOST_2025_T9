# e+BOOST_2025_T9
TB 2025 @ CERN PS T9

At the moment use check_signal_epBOOST.ipynb and compare_epBOOST.ipynb

the first one shows the signal and some correlations for each **run_number**
the second one compare the LG Ph, the Nclu and the Qtot of the different runs in **run_list**

To run the data mount the folder via sshfs running this code in the data directory (it has to be empty to run like this): 
sshfs -o ro yourname@lxplus.cern.ch:/eos/project/i/insulab-como/testBeam/TB_2025_06_T9_epBOOST/HDF5/ .
