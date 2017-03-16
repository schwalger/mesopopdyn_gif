#! /usr/bin/env python


# create parameter file for c code,
# one can also adapt param.h manually
#
# January 2016, Tilo Schwalger, tilo.schwalger@epfl.ch


from pylab import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import multipop16 as mp
import ccodegen


K = 2
size = 200
N = np.array([ 4, 1 ]) * size

mu=np.array([24., 24.])
c=[10., 10.]
Vreset=[0.,0.]  #only for GLIF mode
Jref=[15.,15.]      #only for GLM mode
Vth=15.
DeltaV=2.5
delay=0.001
t_ref=0.004
taum=np.array([0.02,0.02])
taua=[[0.],[0.]]
Ja=[[0.],[0.]]


#connectivity
J = 0.3
g = 5.
pconn = 0.2

N0mat=np.array([[ 800, 200 ], [800, 200]], dtype=float)
Nmat=np.vstack((N,N))

C0 = np.array([[ 800, 200 ], [800, 200]]) * 0.2
C = np.vstack((N,N)) * pconn

Js = np.array([[ J, -g * J], [J, -g * J]]) * C0 / C

taus1_ = [0.003, 0.006]
taus1 = 1 * np.array([taus1_ for k in range(K)])


p1 = mp.MultiPop(N=N, rho_0= c, tau_m = taum, tau_sfa=taua, J_syn=Js, taus1=taus1,delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref, V_reset= np.array(Vreset), J_a=Ja, pconn=pconn, mu= np.array(mu), delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K))

ccodegen.generate_paramfile(p1,fname='param.h')
