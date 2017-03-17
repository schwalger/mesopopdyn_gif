This project provides simulation code associated with the arxiv preprint

[T. Schwalger, M. Deger, and W. Gerstner. Towards a theory of cortical columns: From spiking neurons to interacting neural populations of finite size. ArXiv e-prints, November 2016.](https://arxiv.org/abs/1611.00294)

The core simulation code is written in C. The C libraries can be used either via a Python interface or directly in a C program.


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

*For running a simulaion of an excitatory-inhibitory network (Fig.5 of the paper), change to EI_net_py/ and run

python ei_net_N000_p0.2.py

*For running a simulaion of a population of adapting neurons, change to uncoupled_py/ and run

python lif_adap.py






