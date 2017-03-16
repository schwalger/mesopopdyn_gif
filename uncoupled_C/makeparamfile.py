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


K = 1
mu=np.array([18.])
c=[10.]
Vreset=[0.]  #only for GLIF mode
Jref=0.      #only for GLM mode
Vth=15.
DeltaV=2.
delay=0.001
t_ref=0.004
Js=np.array([[0.]])
taus1=[[0.]]
pconn=np.ones((K,))
step=0.*np.array([[15.]])
tstep=np.array([[0.7]])
taum=np.array([0.02])
N=[1000]
taua=1.
Ja=0.

p1 = mp.MultiPop(N=N, rho_0= c, tau_m = taum, tau_sfa=[[taua]], J_syn=Js, taus1=taus1,delay=np.ones((K,K))*delay, t_ref= np.ones(K)*t_ref, V_reset= np.array(Vreset), J_a=[[Ja]], pconn=pconn, mu= np.array(mu), delta_u= DeltaV*np.ones(K), V_th= Vth*np.ones(K),sigma=np.zeros(K),)

ccodegen.generate_paramfile(p1,fname='param.h')
