#! /usr/bin/env python


# simulate uncoupled population with adaptation (mimicking "pyramidal cell")
# like in Fig.4 of arxiv paper
# January 2016, Tilo Schwalger, tilo.schwalger@epfl.ch


from pylab import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import multipop16 as mp
import multiprocessing as multproc


#integration time step for microscopic and mesoscopic simulation
dt=0.0002
dtpop=0.001

#coarse-grained time step at which activity is recorded (binning of spikes)
dtbin=0.001
dtbinpop=dtbin

#parameters
M = 1                      #number populations
N=[500]
mu=np.array([12.])         
taum=np.array([0.02])
c=[10.]
Vreset=[25.]               #reset potential, not needed for GLM mode 
Vth=10.
Delta_u=2.
delay=0.001
t_ref=0.004
Js=np.array([[0.]])
taus1=[[0.]]
pconn=0.*np.ones((M,))
tau_theta=[[0.01,1.]]
J_theta=[[1.5,1.5]]
mode='glif'                #'glif','glm','glif_master','glm_master'





p1 = mp.MultiPop(dt=dt, N=N, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1,delay=np.ones((M,M))*delay, t_ref= np.ones(M)*t_ref, V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu), delta_u= Delta_u*np.ones(M), V_th= Vth*np.ones(M),sigma=np.zeros(M), mode=mode)
p1.dt_rec=dtbin
p2 = mp.MultiPop(dt=dtpop, N=N, rho_0= c, tau_m = taum, tau_sfa=tau_theta, J_syn=Js, taus1=taus1, delay=np.ones((M,M))*delay, t_ref= np.ones(M)*t_ref, V_reset= np.array(Vreset), J_a=J_theta, pconn=pconn, mu= np.array(mu), delta_u= Delta_u*np.ones(M), V_th= Vth*np.ones(M),sigma=np.zeros(M), mode=mode)
p2.dt_rec=dtbinpop


p1.build_network_tilo_neurons(Nrecord=[1],Vspike=90)
p2.build_network_tilo_populations()





################################################################################
# Plot trajectory
################################################################################

t0=0.4
tend=1.5
i0=int(t0/dtbin)

#step current input
step=[[15.]]   #jump size of mu
tstep=np.array([[0.7]]) #times of jumps

seed=21#20


figure(1)
clf()
subplot(3,1,2)
p1.simulate(tend,step=step,tstep=tstep,seed=seed)
p1.save_trajec()
plot(p1.sim_t[i0:]-p1.sim_t[i0], p1.sim_A[i0:,::-1],color='k',label=r'$A_N$')
ylabel('population activity [Hz]')
legend(loc=2,frameon=False)
ylim((0,90))
xlim((0,tend-t0))
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
ax.set_yticks(arange(0,90,40))
ax.set_yticks(arange(0,90,20),minor=True)
ax.tick_params(which='both',bottom='off', top='off', right='off',left='on', labelleft='on',labelbottom='off')

subplot(3,1,3)
p2.simulate(tend,step=step,tstep=tstep,seed=seed)
p2.save_trajec()
plot(p2.sim_t[i0:]-p2.sim_t[i0], p2.sim_A[i0:],color='b',label=r'$A_N$')
plot(p2.sim_t[i0:]-p2.sim_t[i0], p2.sim_a[i0:,::-1],color='m',label=r'$\bar{A}$')
xlabel('time [s]')
legend(loc=2,frameon=False)
xlim((0,tend-t0))
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_yticks(arange(0,90,40))
ax.set_yticks(arange(0,90,20),minor=True)
# if ax.get_ylim()[1]>100:
#     ylim(ymax=100.)
ylim((0,90))
#p2.clean_trajec()

subplot(3,1,1,frameon=False)
k=0
offset=50.
Nbin=len(p1.voltage[k])
t=p1.dt_rec*np.arange(Nbin)
offset_matrix=np.outer(np.ones(Nbin),np.arange(p1.Nrecord[k])) * offset
plot(t[i0:]-t[i0],p1.voltage[k][i0:]+offset_matrix[i0:],color='k')
xlim((0,tend-t0))
ylim(ymin=-10)
ylabel('V [mV]')
# ax=gca()
# tick_params(bottom='off', top='off', right='off',left='off', labelleft='off',labelbottom='off')
axis('off')
#draw()

#plot step current
axes((0.18,0.93,0.72,0.06),frameon=True)
t=p1.sim_t[i0:]-p1.sim_t[i0]
x=np.ones(len(t))
#plot(t,(t>=tstep[0,0]-p2.sim_t[i0])*(t<tstep[0,1]-p2.sim_t[i0])*x ,color='k')
plot(t,(t>=tstep[0,0]-p1.sim_t[i0])*x ,color='k')
xlim((0,tend-t0))
ylim((-0.2,1.2))
ylabel('I(t)')
axis('off')

fname='trajecs'+p1.__get_parameter_string__() + '.svg'
#savefig(fname)
show()





###################
# print rate and CV
###################

p1.get_isistat()
print 'rate= ',p1.rate
print 'CV= ',p1.cv
p1.save_isih()



###################
# PSD plot
###################

df=0.05
Ntrials=10
nproc=4
dpoints=41 #reduce num of frequency bins to small number  to get a smoothed psd

p1.get_psd(df=df, Ntrials=Ntrials, nproc=nproc,dpoints=dpoints)
p1.save_psd()
p2.get_psd(df=df, Ntrials=Ntrials, nproc=nproc,dpoints=dpoints)
p2.save_psd()


figure(2)
clf()

ax1=subplot(2,1,1)
loglog(p1.freq_log[0], p1.psd_log[0],'o',color='k',ms=4,mfc='None',mec='k',label=r'microsopic')
p2.save_psd()
loglog(p2.freq_log[0], p2.psd_log[0],color='b',label=r'mesoscopic')
setp(ax1,xticklabels=[])
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
xlim((0.05,500))
legend(loc=4,frameon=False)




p1.mu=np.array(mu)+np.array(step)[0,0]
p2.mu=np.array(mu)+np.array(step)[0,0]

p1.build_network_tilo_neurons(Nrecord=[5],Vspike=40)
p2.build_network_tilo_populations()

p1.get_psd(df=df, Ntrials=Ntrials, nproc=nproc,dpoints=dpoints)
p1.save_psd()
p2.get_psd(df=df, Ntrials=Ntrials, nproc=nproc,dpoints=dpoints)
p2.save_psd()


figure(2)
ax2=subplot(2,1,2)
loglog(p1.freq_log[0], p1.psd_log[0],'o',color='k',ms=4,mfc='None',mec='k',label=r'micro, sub')
loglog(p2.freq_log[0], p2.psd_log[0],color='b',label=r'meso, sub')
xlabel('f [Hz]')
ylabel('power spectrum [Hz]')
#setp(ax2,yticklabels=[])
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
xlim((0.05,500))
#savefig('psd'+p1.__get_parameter_string__()+'.svg')
show()




###################
# print rate and CV after step increase to high rates
###################

p1.get_isistat()
print 'rate= ',p1.rate
print 'CV= ',p1.cv
p1.save_isih()
