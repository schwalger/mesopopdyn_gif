**NOTE: There is an updated version of the simulation code as well as novel implementations in the Python and Julia languages [here](https://github.com/schwalger/mesopopdyn_microcircuit). It implements, as an example, the cortical microcircuit model of Potjans and Diesmann, Cereb. Cortex (2014).**

This project provides simulation code associated with the publication

[T. Schwalger, M. Deger, and W. Gerstner. Towards a theory of cortical columns: From spiking neurons to interacting neural populations of finite size. PLoS Comput. Biol., 13(4):e1005507, 2017.](https://infoscience.epfl.ch/record/227465?ln=en)

The core simulation code is written in C. There are two libraries:
- glm_netw_sim_0.8.c for the microscopic (neuron-based) simulation
- glm_popdyn_1.1.c for the mesoscopic (population-based) simulation

The C libraries can be used either via a Python interface or directly in a C program.


Prerequisites for compiling c code
----------------------------------

- gcc
- fftw3 (libfftw3-3 package in ubuntu)
- gsl (libgsl2 package in ubuntu)



Create python modules from c libraries
--------------------------------------

In the top level do:

make -Bf Makefile_fasthazard

This is a modification of the original pseudocode implementation in the paper by using a lookup table for calculating the hazard function. This yields slightly better performance. For the original implementation w/o lookup table use

make -Bf Makefile_orig


Examples
--------

- For running a simulation of an excitatory-inhibitory network (both microscopic and mesoscopic, as in Fig.5 of the paper), change to EI_net_py/ and run

python ei_net_N000_p0.2.py

- For running a simulaion of a population of adapting neurons, change to uncoupled_py/ and run

python lif_adap.py






