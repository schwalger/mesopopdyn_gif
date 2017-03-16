- create data folder if not exist: mkdir data
- set parameters in makeparamfile.py
- python makeparamfile.py
- set time step, time resolution dtbin, simulation time, simulation mode (GLIF or GLM) in Makefile
- compile:
   make netwtrajec_lif  (for microscopic simulation) 
   make poptrajec_lif   (for mesoscopic simulation)
- ./a.out
- data (activity vs time) stored in data folder



