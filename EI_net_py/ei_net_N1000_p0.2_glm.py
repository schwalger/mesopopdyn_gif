#! /usr/bin/env python


# simulate trajectories and psd for EI network
# same as Fig.5D,E in arxiv paper but in GLM mode (no reset to fixed voltage but refractory kernel of amplitude Jref)
#
# April 2016, Tilo Schwalger, tilo.schwalger@epfl.ch


from pylab import *
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import multipop16 as mp
import time
import matplotlib.gridspec as gridspec

with_psd = True
with_micro = True
#with_psd = False
#with_micro = False


dt=0.0002
dtbin=0.0002
dtpop=0.0002
dtbinpop=dtbin
t0=1.
tend=t0+1.

#Parameters
K = 2
c = np.ones(K, dtype=float) * 10.
Jref = [15. for k in range(K)]   # in GLM mode amount of instantaneous decrease of membrn pot at spike, equivalent to kernel increasing threshold with initial amplitude of Jref mV
Vth =    [15. for k in range(K)]
delay = 0.001#0.001
t_ref = 0.004
taum = [0.02 for k in range(K)]
size = 200
N = np.array([ 4, 1 ]) * size
DeltaV = 2.5
mu = 24. * np.ones(K)

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

tau_sfa_exc = [0.1, 0.3, 1.]#[0.05,0.1, 0.3,0.65] #time scales of kernels additional to refractory kernel
tau_sfa_inh = [0.05]
J_sfa_exc = [0.,0.,0.]
J_sfa_inh = [0.]
tau_theta = [tau_sfa_exc, tau_sfa_inh]
J_theta =   [J_sfa_exc,   J_sfa_inh  ]
mode='glm'
step=0









#Building the networks
if with_micro:
    p1 = mp.MultiPop(dt=dt, N=N, rho_0=c, tau_m = taum, tau_sfa=tau_theta,\
                         J_syn=Js, taus1=taus1, delay=np.ones((K,K))*delay, \
                         t_ref= np.ones(K)*t_ref,  Jref=Jref, J_a=J_theta,\
                         pconn=pconn, mu=np.array(mu), delta_u= DeltaV*np.ones(K), \
                         V_th=  np.array(Vth),sigma=np.zeros(K), mode=mode)
    p1.dt_rec=dtbin
    p1.build_network_tilo_neurons(Nrecord=[5, 0],Vspike=90)
# p1.build_network_tilo_neurons()


p2 = mp.MultiPop(dt=dtpop, N=N, rho_0=c, tau_m = taum, tau_sfa=tau_theta,\
     J_syn=Js, taus1=taus1, delay=np.ones((K,K))*delay, \
     t_ref= np.ones(K)*t_ref,Jref=Jref, J_a=J_theta, \
     pconn=pconn, mu=np.array(mu), delta_u= DeltaV*np.ones(K), \
     V_th=  np.array(Vth),sigma=np.zeros(K), mode=mode)
p2.dt_rec=dtbinpop
p2.build_network_tilo_populations()


#TRAJECTORIES

if with_micro:
    p1.simulate(tend)

p2.simulate(tend)
p2.save_trajec()

i0=int(t0/dtbin)

rcParams['legend.fontsize']= 'medium'
rcParams['legend.frameon']= False
#subplots_adjust(top=0.87,bottom=0.15)


figure(2, figsize=(2.5,2.5*1.),dpi=400)
clf()

ymax=100.
major_tick_spacing=50.
minor_tick_spacing=25.

gs = gridspec.GridSpec(3, 1, height_ratios=[4,2,2])

if with_micro:
    ax2 = subplot(gs[1])
    plot(p1.sim_t[i0:]-p1.sim_t[i0], p1.sim_A[i0:,0],color='sage',lw=1.,label=r'$A_N$')
    ylabel('population activity [Hz]')
    legend(loc=2,frameon=False,bbox_to_anchor=(0.,1.2))
    ylim((0,ymax))
    xlim((0,tend-t0))
    ax=gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    ax.set_yticks(arange(0,ymax+0.5,major_tick_spacing))
    ax.set_yticks(arange(0,ymax+0.5,minor_tick_spacing),minor=True)
    ax.tick_params(which='both',bottom='off', top='off', right='off',left='on', labelleft='on',labelbottom='off')


ax2 = subplot(gs[2])
plot(p2.sim_t[i0:]-p2.sim_t[i0], p2.sim_A[i0:,0],color='darkgreen',lw=1,label=r'$A_N$')
plot(p2.sim_t[i0:]-p2.sim_t[i0], p2.sim_a[i0:,0],color='lightgreen',lw=1,label=r'$\bar{A}$')
xlabel('time [s]')
legend(loc=2,frameon=False,bbox_to_anchor=(0.,1.3),ncol=2)
xlim((0,tend-t0))
ax=gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_yticks(arange(0,ymax+0.5,major_tick_spacing))
ax.set_yticks(arange(0,ymax+0.5,minor_tick_spacing),minor=True)
# if ax.get_ylim()[1]>100:
#     ylim(ymax=100.)
ylim((0,ymax))


if with_micro:
#subplot(3,1,1,frameon=False)
    ax2 = subplot(gs[0])
    k=0
    offset=120.
    Nbin=len(p1.voltage[k])
    t=p1.dt_rec*np.arange(Nbin)
    offset_matrix=np.outer(np.ones(Nbin),np.arange(p1.Nrecord[k])) * offset
    plot(t[i0:]-t[i0],p1.voltage[k][i0:]+offset_matrix[i0:],color='k', lw=0.5)
    xlim((0,tend-t0))
    ylim(ymin=-10)
    ylabel('V [mV]')
    # ax=gca()
    # tick_params(bottom='off', top='off', right='off',left='off', labelleft='off',labelbottom='off')
    axis('off')
#draw()







fname='trjc_ei_N%d_p%g'%(N[0]+N[1],pconn)
# savefig(fname+'.svg')
# savefig(fname+'.png')
show()







###################
# print rate and CV
###################

if with_micro:
    p1.get_isistat()
    print 'rate= ',p1.rate
    print 'CV= ',p1.cv
    p1.save_isih()



if not with_psd:
    sys.exit()


###################
# PSD plot
###################

Ntrials=32
df=1.
nproc=4

if with_micro:
    p1.get_psd(df=df, Ntrials=Ntrials, nproc=nproc,dpoints=101)
    p1.save_psd()

p2.get_psd(df=df, Ntrials=Ntrials, nproc=nproc,dpoints=201)
p2.save_psd()


figure(3)
clf()
if with_micro:
#    psd = (N[0] * p1.psd_log[0] + N[1] * p1.psd_log[1]) / (N[0]+N[1])
    psd = p1.psd_log[0]
    loglog(p1.freq_log[0], psd,'o',color='k',mfc='None',mec='k')

#psd = (N[0] * p2.psd_log[0] + N[1] * p2.psd_log[1]) / (N[0]+N[1])
psd = p2.psd_log[0]
loglog(p2.freq_log[0], psd,'-b')
ax1=gca()
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(which='both',bottom='on', top='off', right='off',left='on', labelleft='on',labelbottom='on')
xlim(xmax=500)
xlabel('f [Hz]')
ylabel('power spectrum [Hz]')
#legend(loc=4,bbox_to_anchor=(1,-0.1))
show()



