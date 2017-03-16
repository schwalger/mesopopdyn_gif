# change to multipop15.py: based on multipop_c_13
#
#
# Moritz Deger, moritz.deger@epfl.ch, May 15, 2015
# Tilo Schwalger, tilo.schwalger@epfl.ch, May 15, 2015

import pdb

import numpy as np
try:
   import pandas
except:
   print 'failed to import pandas'
import pylab
import os.path
import os

#import nest    #for use of NEST simulator
import multipop_c_13 as mpc #custom-made C simulation of networks and populations
from numpy.fft import fft


class MultiPop(object):

    def __init__(self, dt=0.0002, N=[400,100], rho_0=[10.,5.], \
                 tau_m=[0.01,0.01], tau_sfa=[[3.],[1.]], \
                 J_syn=[[0.05,-0.22],[0.05,-0.22]], delay=[[0.002,0.002],[0.002,0.002]], \
                 t_ref=[0.002,0.002], V_reset=[0.,0.], J_a=[[1.],[0.0]], \
                 pconn=np.ones((2,2))*1., \
                 mu=[0.,0.], delta_u=[4.,4.], V_th=[0.,0.], \
                 taus1=[[0,0.],[0.,0.]],\
                 taur1=[[0,0.],[0.,0.]],\
                 taus2=[[0,0.],[0.,0.]],\
                 taur2=[[0,0.],[0.,0.]],\
                 a1=[[1.,1.],[1.,1.]],\
                 a2=[[0,0.],[0.,0.]], Jref=[10.,10.],sigma=[0.,0.], mode='glif' \
            ):
        self.dt = dt
        self.N = np.array(N)
        self.K = len(self.N)
        self.rho_0 = np.array(rho_0)
        self.tau_m = np.array(tau_m)
        self.tau_sfa = tau_sfa
        self.delay = np.array(delay)
        self.t_ref = np.array(t_ref)
        self.V_reset = np.array(V_reset)
        self.V_th = np.array(V_th)
        self.J_a = J_a
        self.J_syn = np.array(J_syn)
        self.pconn = np.resize(pconn, (self.K,self.K))
        self.delta_u = delta_u
        self.mu = np.array(mu) 
        self.taus1=np.resize(taus1,(self.K,self.K))
        self.taur1=np.resize(taur1, (self.K,self.K))
        self.taus2=np.resize(taus2, (self.K,self.K))
        self.taur2=np.resize(taur2, (self.K,self.K))
        self.a1=np.resize(a1, (self.K,self.K))
        self.a2=np.resize(a2, (self.K,self.K))
        self.sigma=np.array(sigma)
        if mode=='glif':
            self.mode=10
            self.Jref=np.zeros(self.K)
        elif mode=='glif_master':
            self.mode=30
            self.Jref=np.zeros(self.K)
        elif mode=='glif4':
            self.mode=14
            self.Jref=np.zeros(self.K)
        elif mode=='glm':
            self.mode=0
            self.Jref=np.array(Jref)
        elif mode=='glif_naiv':
            self.mode=12
            self.Jref=np.zeros(self.K)
        elif mode=='glm_naiv':
            self.mode=2
            self.Jref=np.array(Jref)
        elif mode=='glm_master':
            self.mode=20
            self.Jref=np.array(Jref)
        else:
            self.mode=10
            self.Jref=np.zeros(self.K)
            
            
        assert(self.J_syn.shape==(self.K,self.K))
        
        # non-exposed but settable parameters
        self.len_kernel = -1    # -1 triggers automatic history size
        self.local_num_threads = 1 #2
        self.dt_rec = self.dt
        self.n_neurons_record_rate = 10
        self.origin=0. #time origin
        self.step=None
        self.tstep=None
        
        # internal switches
        self.__sim_mode__ = None
        
        
    def __build_network_common__(self):
        nest.set_verbosity("M_WARNING")
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': self.dt*1e3, 'print_time': True, \
            'local_num_threads':self.local_num_threads})
        
        self.tau_1_ex = np.ones(self.K) * np.nan
        self.tau_1_in = np.ones(self.K) * np.nan
        self.tau_2_ex = np.ones(self.K) * np.nan
        self.tau_2_in = np.ones(self.K) * np.nan
        for i in range(self.K):
            for j in range(self.K):
                if self.J_syn[i,j]>=0:
                    if np.isnan( self.tau_1_ex[i] ):
                        self.tau_1_ex[i] = self.taus1[i,j]
                    else:
                        assert( np.allclose( self.tau_1_ex[i], self.taus1[i,j] ) )
                    if np.isnan( self.tau_2_ex[i] ):
                        self.tau_2_ex[i] = self.taur1[i,j]
                    else:
                        assert( np.allclose( self.tau_2_ex[i], self.taur1[i,j] ) )
                else:
                    if np.isnan( self.tau_1_in[i] ):
                        self.tau_1_in[i] = self.taus1[i,j]
                    else:
                        assert( np.allclose( self.tau_1_in[i], self.taus1[i,j] ) )
                    if np.isnan( self.tau_2_in[i] ):
                        self.tau_2_in[i] = self.taur1[i,j]
                    else:
                        assert( np.allclose( self.tau_2_in[i], self.taur1[i,j] ) )

        for i in range(self.K):
           if np.isnan( self.tau_1_ex[i] ):
              self.tau_1_ex[i] = 1e-8
           if np.isnan( self.tau_1_in[i] ):
              self.tau_1_in[i] = 1e-8
           if np.isnan( self.tau_2_ex[i] ):
              self.tau_2_ex[i] = 1e-8
           if np.isnan( self.tau_2_in[i] ):
              self.tau_2_in[i] = 1e-8

        #if time constants is zero, set it to 1e-8
        for i in range(self.K):
            if (self.tau_1_ex[i]==0):
                self.tau_1_ex[i]=1e-8
            if (self.tau_2_ex[i]==0):
                self.tau_2_ex[i]=1e-8
            if (self.tau_1_in[i]==0):
                self.tau_1_in[i]=1e-8
            if (self.tau_2_in[i]==0):
                self.tau_2_in[i]=1e-8
        
        # we need to set C_m for nest, but it is divided out later again.
        self.C_m = np.ones(self.K) * 250.
        
        # determine rescaling of synapses for nest
        self.J_syn_nestfactor = np.ones_like(self.J_syn)
        for i in range( self.J_syn.shape[1] ):
            if (self.J_syn[:,i] >= 0).all():
                self.J_syn_nestfactor[:,i] = self.C_m / (self.tau_1_ex[i] * 1e3 )
            elif (self.J_syn[:,i] <= 0).all():
                self.J_syn_nestfactor[:,i] = self.C_m / (self.tau_1_in[i] * 1e3 )
            else:
                # this must not happen, because populations are either ex or in but not both
                assert(False)
    
    
    def build_network_populations(self):
        self.__build_network_common__()
        self.nest_pops = nest.Create('gif_pop_psc_exp', self.K)
        
        # set single neuron properties
        for i, nest_i in enumerate( self.nest_pops ):
            nest.SetStatus([nest_i], {
                'C_m': self.C_m[i],
                'I_e': self.mu[i] / (self.tau_m[i] * 1e3 / self.C_m[i]),
                'lambda_0': self.rho_0[i],
                'Delta_V': self.delta_u[i],
                'tau_m': self.tau_m[i] *1e3,
                'tau_sfa': np.array(self.tau_sfa[i]) * 1e3,
                'q_sfa': np.array(self.J_a[i]) / np.array(self.tau_sfa[i]),
                'V_T_star': self.V_th[i],
#                'V_reset': self.V_th[i]-self.V_reset[i],
                'V_reset': self.V_reset[i],
                'len_kernel': self.len_kernel,
                'N': self.N[i],
                't_ref': self.t_ref[i]*1e3,
                'tau_syn_ex': max([self.tau_1_ex[i] * 1e3, 0.1]),
                'tau_syn_in': max([self.tau_1_in[i] * 1e3, 0.1]),
                'E_L': 0.
                })
        # beta synapses are not supported
        assert( np.allclose( [self.tau_2_ex[i], self.tau_2_in[i]], 0. ) )
        
        # connect the populations
        for i, nest_i in enumerate( self.nest_pops ):
            for j, nest_j in enumerate( self.nest_pops ):
                nest.SetDefaults('static_synapse', {
                    'weight': self.J_syn[i,j] * \
                        self.J_syn_nestfactor[i,j], #/ float(self.N[j])
                    'delay': self.delay[i,j]*1e3} )
                nest.Connect( [nest_j], [nest_i], 'all_to_all')
        
        # monitor the output using a multimeter, this only records with dt_rec!
        self.nest_mm = nest.Create('multimeter')
        nest.SetStatus( self.nest_mm, {'record_from':['n_events', 'mean'],
            'withgid': True, 'withtime': False, 'interval': self.dt_rec*1e3})
        nest.Connect(self.nest_mm, self.nest_pops, 'all_to_all')
        
        # monitor the output using a spike detector
        self.nest_sd = []
        for i, nest_i in enumerate( self.nest_pops ):
          self.nest_sd.append( nest.Create('spike_detector') )
          nest.SetStatus( self.nest_sd[i], {'withgid': False, 
                                            'withtime': True, 'time_in_steps': True})
          nest.SetDefaults('static_synapse', {
             'weight': 1.,
             'delay': self.dt*1e3} )
          nest.Connect( [self.nest_pops[i]], self.nest_sd[i], 'all_to_all')

        # save information that we are in population mode
        self.__sim_mode__ = 'populations'
        


    def build_network_neurons(self, record_rate=False, Nrecord=None):
       self.__build_network_common__()
       self.record_rate = record_rate
       self.Nrecord = Nrecord
       self.nest_pops = []
       for k in range(self.K):
          self.nest_pops.append( nest.Create('gif_psc_exp', self.N[k]) )
        
       with_reset = self.mode>=10
       # the model gif_psc_exp does not support with_reset=False.
       assert(with_reset)
        
       # set single neuron properties
       for i, nest_i in enumerate( self.nest_pops ):
          nest.SetStatus(nest_i, {
             'C_m': self.C_m[i],
             'I_e': self.mu[i] / (self.tau_m[i] * 1e3 / self.C_m[i]),
             'lambda_0': self.rho_0[i],
             'Delta_V': self.delta_u[i],
             'g_L' : self.C_m[i] / (self.tau_m[i] * 1e3 ),
             'tau_sfa': np.array(self.tau_sfa[i]) * 1e3,
             'q_sfa': np.array(self.J_a[i]) / np.array(self.tau_sfa[i]), 
             'V_T_star': self.V_th[i],
             'V_reset': self.V_reset[i],
             'tau_syn_ex': max([self.tau_1_ex[i] * 1e3, 0.1]),
             'tau_syn_in': max([self.tau_1_in[i] * 1e3, 0.1]),
             'E_L': 0.,
             't_ref': self.t_ref[i]*1e3,
             'V_m': 0.
          })
       # beta synapses are not supported
       assert( np.allclose( [self.tau_2_ex[i], self.tau_2_in[i]], 0. ) )
        
       # connect the populations
       for i, nest_i in enumerate( self.nest_pops ):
          for j, nest_j in enumerate( self.nest_pops ):
             nest.SetDefaults('static_synapse', {
                'weight': self.J_syn[i,j]  * \
                        self.J_syn_nestfactor[i,j],  
                'delay': self.delay[i,j]*1e3} )
             if np.allclose( self.pconn[i,j], 1. ):
                conn_spec = {'rule': 'all_to_all'}
             else:
                conn_spec = {'rule': 'pairwise_bernoulli', \
                             'p': self.pconn[i,j]}
             print 'connecting population', j, 'to', i
             nest.Connect( nest_j, nest_i, conn_spec )
        
       # monitor the output using a multimeter and a spike detector
       self.nest_sd = []
       for i, nest_i in enumerate( self.nest_pops ):
          self.nest_sd.append( nest.Create('spike_detector') )
          nest.SetStatus( self.nest_sd[i], {'withgid': False, 
                                            'withtime': True, 'time_in_steps': True})
          nest.SetDefaults('static_synapse', {
             'weight': 1.,
             'delay': self.dt*1e3} )
          nest.Connect( self.nest_pops[i], self.nest_sd[i], 'all_to_all')

       if self.record_rate:
          self.nest_mm_rate = []
          for i, nest_i in enumerate( self.nest_pops ):
             self.nest_mm_rate.append( nest.Create('multimeter') )
             nest.SetStatus( self.nest_mm_rate[i], {'record_from':['rate'], \
                                               'withgid': False, 'withtime': True, \
                                               'interval': self.dt_rec*1e3})
             nest.Connect(self.nest_mm_rate[i], list( np.array(self.nest_pops[i])), 'all_to_all')

       if (self.Nrecord!=None):
          self.nest_mm_Vm = []
          for i, nest_i in enumerate( self.nest_pops ):
             self.nest_mm_Vm.append( nest.Create('multimeter') )
             nest.SetStatus( self.nest_mm_Vm[i], {'record_from':['V_m'], \
                                               'withgid': True, 'withtime': True, \
                                               'interval': self.dt_rec*1e3})
             nest.Connect(self.nest_mm_Vm[i], list( np.array(self.nest_pops[i])[:self.Nrecord[i]]), 'all_to_all')


       # save information that we are in neuron mode
       self.__sim_mode__ = 'neurons'


#this function still uses old nest models and is therefore disabled.
#    def update_nest_neuron_params(self, rho_0=[10.,5.], tau_m=[0.01,0.01], tau_sfa=[[3.],[1.]], \
#                                t_ref=[0.002,0.002], V_reset=[0.,0.], J_a=[[1.],[0.0]], \
#                                mu=[0.,0.], delta_u=[4.,4.], V_th=[0.,0.], Jref=[10.,10.], mode='glif'):

#       self.rho_0 = np.array(rho_0)
#       self.tau_m = np.array(tau_m)
#       self.tau_sfa = tau_sfa
#       self.t_ref = np.array(t_ref)
#       self.V_reset = np.array(V_reset)
#       self.V_th = np.array(V_th)
#       self.J_a = J_a
#       self.delta_u = delta_u
#       self.mu = np.array(mu) 
#       if mode=='glif':
#          self.mode=10
#          self.Jref=np.zeros(self.K)
#       if mode=='glif4':
#          self.mode=14
#          self.Jref=np.zeros(self.K)
#       elif mode=='glm':
#          self.mode=0
#          self.Jref=np.array(Jref)
#       elif mode=='glif_naiv':
#          self.mode=12
#          self.Jref=np.zeros(self.K)
#       elif mode=='glm_naiv':
#          self.mode=2
#          self.Jref=np.array(Jref)
#       else:
#          self.mode=10
#          self.Jref=np.zeros(self.K)
#          
#       with_reset = self.mode>=10
##       nest.SetKernelStatus({'time':0.})
#       self.origin=nest.GetKernelStatus('time') * 1e-3
#      
#       # set single neuron properties
#       for i, nest_i in enumerate( self.nest_pops ):
#          nest.SetStatus(nest_i, {
#             'C_m': self.C_m[i],
#             'I_e': self.mu[i] / (self.tau_m[i] * 1e3 / self.C_m[i]),
#             'c_2': self.rho_0[i],
#             'c_3': 1./self.delta_u[i],
#             'tau_m': self.tau_m[i] *1e3,
#             'tau_sfa': np.array(self.tau_sfa[i]) * 1e3,
#             'q_sfa': np.array(self.J_a[i]) / np.array(self.tau_sfa[i]), 
#             'V_th': self.V_th[i],
#             'V_reset': self.V_reset[i],
#             'dead_time': self.t_ref[i]*1e3,
#             'with_reset': with_reset,
#           })
#        
#       # clear recording devices,  multimeter and a spike detector
#       for i, nest_i in enumerate( self.nest_pops ):
#          nest.SetStatus( self.nest_sd[i], {'n_events': 0, 'origin': self.origin * 1e3})

#       if self.record_rate:
#          for i, nest_i in enumerate( self.nest_pops ):
#             nest.SetStatus( self.nest_mm_rate[i], {'n_events': 0, 'origin': self.origin * 1e3})

#       for i, nest_i in enumerate( self.nest_pops ):
#          if self.Nrecord != None:      
#             nest.SetStatus( self.nest_mm_Vm[i], {'n_events': 0, 'origin': self.origin * 1e3})
#            
#       # save information that we are in neuron mode
#       self.__sim_mode__ = 'neurons'


    def __build_network_tilo_common__(self):

        self.mp = mpc.Multipop(self.dt_rec, self.dt, \
                       tref=self.t_ref, taum = self.tau_m, \
                       taus1=self.taus1, taur1=self.taur1, taus2=self.taus2, taur2=self.taur2, a1=self.a1, a2=self.a2, \
                       mu=self.mu, c=self.rho_0, DeltaV=self.delta_u, \
                       delay = self.delay[0], vth=self.V_th, vreset=self.V_reset, N=self.N, \
                       J = self.J_syn, \
                       p_conn=np.ones((self.K, self.K)) * self.pconn,\
                                    Jref=self.Jref, J_theta= self.J_a, tau_theta= self.tau_sfa, sigma=self.sigma, mode=self.mode)


    def build_network_tilo_populations(self):

        # #build corresponding fully-connected network
        # self.J_syn *= self.pconn
        # self.pconn=np.ones((self.K, self.K))

        self.__build_network_tilo_common__()
        
        # save information that we are in population mode
        self.__sim_mode__ = 'pop_tilo'

    def build_network_tilo_neurons(self, Nrecord=[0,0],Vspike=30.):
        self.__build_network_tilo_common__()
        
        if (sum(Nrecord)==0):
            # save information that we are in neuron mode
            self.__sim_mode__ = 'netw_tilo'
        else:
            self.__sim_mode__ = 'netw_tilo_record_voltage'
            self.Nrecord=np.array(Nrecord)
            self.Nrecord.resize(self.K,refcheck=False)  #fills with zeros if less elemens than number of populations
            self.Vspike=Vspike

        



    def __debug_record_nest_state_variables__(self):
        self.nest_debug_mm = []
        if self.__sim_mode__=='neurons':
            for i,pop in enumerate(self.nest_pops):
                self.nest_debug_mm.append( nest.Create('multimeter') )
                nest.SetStatus(self.nest_debug_mm[-1], {\
                    'record_from':['rate', 'V_m', 'E_sfa', 'I_syn_ex', \
                    'I_syn_in'], 'withgid': True, 'withtime': True, \
                    'interval': self.dt*1e3})
                nest.Connect( self.nest_debug_mm[-1], pop, 'all_to_all')
        elif self.__sim_mode__=='populations':
            for i,pop in enumerate(self.nest_pops):
                self.nest_debug_mm.append( nest.Create('multimeter') )
                nest.SetStatus(self.nest_debug_mm[-1], {\
                    'record_from':['n_events', 'mean', 'V_m', 'E_sfa', \
                    'I_syn_ex', \
                    'I_syn_in'], 'withgid': True, 'withtime': True, \
                    'interval': self.dt*1e3})
                nest.Connect( self.nest_debug_mm[-1], [pop] )
    
    
    def retrieve_sim_data(self):
        if self.__sim_mode__=='populations':
            self.retrieve_sim_data_populations()
        elif self.__sim_mode__=='neurons':
            self.retrieve_sim_data_neurons()
        else:
            print 'No network has been built. Call build_network... first!'
        self.rate = self.get_firingrates()
    
    
    def retrieve_sim_data_populations(self):
        assert(self.__sim_mode__=='populations')

        # extract data from multimeter
        data_mm = nest.GetStatus( self.nest_mm )[0]['events']
        for i, nest_i in enumerate( self.nest_pops ):
            ev_i = data_mm['n_events'][ data_mm['senders']==nest_i ]
            a_i  = data_mm['mean'][ data_mm['senders']==nest_i ]
            A = ev_i.astype(float) / self.N[i] / self.dt
            a = a_i / self.N[i] / self.dt
            min_len = np.min([len(a), len(self.sim_a)])
            self.sim_A_mm = np.zeros_like(self.sim_A)
            self.sim_A_mm[:min_len,i] = A[:min_len]
            self.sim_a[:min_len,i] = a[:min_len]
        
            # extract data from spike detector and bin to dt_rec
            data_sd = nest.GetStatus(self.nest_sd[i], \
                keys=['events'])[0][0]['times'] * self.dt - self.origin
            bins = np.concatenate((self.sim_t, \
                np.array([self.sim_t[-1]+self.dt_rec])))
            A = np.histogram(data_sd, bins=bins)[0] / \
                float(self.N[i]) / self.dt_rec
#            self.sim_A[:min_len,i] = A[:min_len]
            self.sim_A[:,i]=A
    

    def retrieve_sim_data_neurons(self):
        assert(self.__sim_mode__=='neurons')
        for i, nest_i in enumerate( self.nest_pops ):
            if self.record_rate:
               data_mm = get_dataframe( self.nest_mm_rate[i][0] )
               a = data_mm.groupby('times').mean().rate
               min_len = np.min([len(a), len(self.sim_a)])
               self.sim_a[:min_len,i] = np.array(a)[:min_len]

            data_sd = nest.GetStatus(self.nest_sd[i], \
                keys=['events'])[0][0]['times'] * self.dt - self.origin
            bins = np.concatenate((self.sim_t, \
                np.array([self.sim_t[-1]+self.dt_rec])))
            A = np.histogram(data_sd, bins=bins)[0] / \
                float(self.N[i]) / self.dt_rec
#            self.sim_A[:min_len,i] = A[:min_len]
            self.sim_A[:,i]=A

        if (self.record_rate==False):
           self.sim_a=self.__moving_average__(self.sim_A.T,int(0.05/self.dt_rec)).T 

        if self.Nrecord!=None:
            self.voltage = np.array([nest.GetStatus( self.nest_mm_Vm[i] )\
                [0]['events']['V_m'] for i in range(self.K)])


    def __moving_average__(self,a, n=3) :
        """
        computes moving average with time window of length n along the second axis
        """
        print 'compute moving average with window length', n
        ret = np.cumsum(a, axis = 1, dtype=float)
        ret[:,n:] = ret[:,n:] - ret[:,:-n]
        return ret/ n 
  



    def simulate(self, T,step=None,tstep=None,seed=365, seed_quenched=1, ForceSim=False):
        self.sim_T = T
        self.seed=seed
        if step is None:
           self.step=None
           self.tstep=None         
        else: 
           self.step=np.array(step)
           self.tstep=np.array(tstep)
         
        fname=self.__trajec_name__()
        #print fname
        if os.path.exists(fname) and not ForceSim:
            print 'load existing trajectories'
            print fname
            f=np.load(fname)
            self.sim_t=f['t']
            self.sim_a=f['a']
            self.sim_A=f['A']
            if self.__sim_mode__=='netw_tilo_record_voltage':
                self.voltage=[np.vstack(f['V'][i]) for i in range(self.K)]
                self.threshold=[np.vstack(f['theta'][i]) for i in range(self.K)]
        else:
            if self.__sim_mode__==None:
                print 'No network has been built. Call build_network... first!'

            elif self.__sim_mode__=='pop_tilo':
                self.mp.get_trajectory_pop(self.sim_T,step,tstep,seed=seed)
                self.sim_A=self.mp.A.T
                self.sim_a=self.mp.a.T  
                self.sim_t = self.dt_rec * np.arange(len(self.sim_A))

            elif self.__sim_mode__=='netw_tilo':
                self.mp.get_trajectory_neuron(self.sim_T,step,tstep,seed=seed, seed_quenched=seed_quenched)
                self.sim_A=self.mp.A.T
                self.sim_a=self.__moving_average__(self.mp.A,int(0.05/self.dt_rec)).T 
                self.sim_t = self.dt_rec * np.arange(len(self.sim_A))

            elif self.__sim_mode__=='netw_tilo_record_voltage':
                self.mp.get_trajectory_voltage_neuron(self.sim_T,self.Nrecord, self.Vspike,step,tstep,seed=seed, seed_quenched=seed_quenched)
                #transpose data such that the 1st axis refers to time, 2nd axis is population or neuron, respectively
                self.sim_A=self.mp.A.T
                self.sim_a=self.__moving_average__(self.mp.A,int(0.05/self.dt_rec)).T
                self.voltage=[v.T for v in self.mp.voltage]
                self.threshold=[theta.T for theta in self.mp.threshold]
                self.sim_t = self.dt_rec * np.arange(len(self.sim_A))

            else:
                # msd =self.local_num_threads * seed + 1 #master seed
                # nest.SetKernelStatus({'rng_seeds': range(msd, msd+self.local_num_threads)})
                # print nest.GetKernelStatus('rng_seeds')
                self.sim_t = np.arange(0., self.sim_T, self.dt_rec)
                self.sim_A = np.ones( (self.sim_t.size, self.K) ) * np.nan
                self.sim_a = np.ones_like( self.sim_A ) * np.nan
                
                if (step!=None):
                    #set initial value (at t0+dt) of step current generator to zero
                    t0=self.origin * 1e3
                    tstep = np.hstack((self.dt * np.ones((self.K,1)), self.tstep)) * 1e3
                    step =  np.hstack((np.zeros((self.K,1)), self.step))
                    # create the step current devices if they do not exist already
                    if not self.__dict__.has_key('nest_stepcurrent'):
                        self.nest_stepcurrent = nest.Create('step_current_generator', self.K )
                    # set the parameters for the step currents
                    for i in range(self.K):
                        nest.SetStatus( [self.nest_stepcurrent[i]], {
                           'amplitude_times': tstep[i] + t0,
                           'amplitude_values': step[i] / (self.tau_m[i] * 1e3 / self.C_m[i]), 'origin': t0, 'stop': self.sim_T * 1e3#, 'stop': self.sim_T * 1e3 + t0
                           })
                        pop_ = self.nest_pops[i]
                        if type(self.nest_pops[i])==int:
                            pop_ = [pop_]
                        nest.Connect( [self.nest_stepcurrent[i]], pop_, syn_spec={'weight':1.} )

                # simulate 1 step longer to make sure all self.sim_t are simulated
                nest.Simulate( (self.sim_T+self.dt) * 1e3 )
                self.retrieve_sim_data()

    def __rebin_log__(self,f,y,nbin):
       x=np.log10(f)
       df=f[1]-f[0]
       n=nbin+1
       dx=(x[-1]-x[0])/nbin
        
       left=x[0]-0.5*dx + dx*np.arange(nbin)
       right=left+dx
       xc=left+0.5*dx
       count=np.zeros(nbin)
       y_av=np.zeros(nbin)
       for i in range(len(x)):
          indx=int((x[i]-left[0])/dx)
          if indx>=nbin: break
          count[indx]+=1
          y_av[indx]+=y[i]
       for i in range(nbin):
          if count[i]>0:
             y_av[i]=y_av[i]/count[i]
          else:
             y_av[i]=np.nan
       fout=10**(xc[np.where(np.isnan(y_av)==False)])
       yout=y_av[np.where(np.isnan(y_av)==False)]
       return (fout,yout)


 
    def get_psd(self, df=0.1, dt_sample=0.001, Ntrials=10, nproc=1, dpoints=100):
        print ''
        if self.__sim_mode__==None:
           print 'get_psd(): No network has been built. Call build_network... first!'
        elif self.__sim_mode__=='pop_tilo':
           print '+++ GET PSD FROM MESOSCOPIC SIMULATION +++'
        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
           print '+++ GET PSD FROM MICROSCOPIC SIMULATION +++'

        self.Ntrials=Ntrials
        self.df=df

        fname=self.__psd_name__()
        print fname
        if os.path.exists(fname):
            print 'LOAD EXISTING PSD DATA'
            X=np.loadtxt(fname)
            self.freq=X[:,0]
            self.psd=X[:,1:]
            self.freq_log=[]
            self.psd_log=[]
            for i in range(self.K):
               x,y = self.__rebin_log__(self.freq,self.psd[:,i],dpoints)
               self.freq_log.append(x)
               self.psd_log.append(y)
        else:
            if self.__sim_mode__=='pop_tilo':
                self.mp.get_psd_pop(df=df, dt_sample=dt_sample, Ntrials=Ntrials, nproc=nproc)
                
                self.freq_log=[]
                self.psd_log=[]
                for i in range(self.K):
                   x,y = self.__rebin_log__(self.mp.f,self.mp.SA[i],dpoints)
                   self.freq_log.append(x)
                   self.psd_log.append(y)
                self.freq=self.mp.f
                self.psd=self.mp.SA.T #each column corresponds to the psd of one population

            elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):

                self.mp.get_psd_neuron(df=df, dt_sample=dt_sample, Ntrials=Ntrials, nproc=nproc)
                self.freq_log=[]
                self.psd_log=[]
                for i in range(self.K):
                   x,y = self.__rebin_log__(self.mp.f,self.mp.SA[i],dpoints)
                   self.freq_log.append(x)
                   self.psd_log.append(y)
                self.freq=self.mp.f
                self.psd=self.mp.SA.T #each column corresponds to the psd of one population
            elif (self.__sim_mode__=='neurons'):
                self.build_network_neurons() #Reset Nest
                NFFT=int(1./(dt_sample*df)+0.5)
                df=1./(NFFT*dt_sample)
                Ntot=NFFT*Ntrials
                self.simulate(Ntot*dt_sample+0.005) #simulate 5ms more
                print 'rate NEST: ',self.get_firingrates()
                self.freq=[]
                self.psd=[]
                self.freq_log=[]
                self.psd_log=[]

                for i in range(self.K):
                   x=self.sim_A[NFFT:,i]
                   L=len(x)
                   x=x[:(L/NFFT)*NFFT].reshape((-1,NFFT))
                   ntrials=x.shape[0]
                   xF=fft(x)
                   S=np.sum(np.real(xF*xF.conjugate()),axis=0)*dt_sample/(NFFT-1)/ntrials
                   psd=S[1:NFFT/2]
                   freq=df*np.arange(NFFT/2-1)+df
                   f_log,psd_log = self.__rebin_log__(freq,psd,dpoints)
                   self.psd.append(psd)
                   self.freq.append(freq)
                   self.freq_log.append(f_log)
                   self.psd_log.append(psd_log)
                   
                   self.psd=np.array(self.psd).T
                   self.freq=np.array(self.freq[0])
                   
            elif (self.__sim_mode__=='populations'):
                self.build_network_populations() #Reset Nest
                NFFT=int(1./(dt_sample*df)+0.5)
                df=1./(NFFT*dt_sample)
                Ntot=NFFT*Ntrials
                self.simulate(Ntot*dt_sample+0.005) #simulate 5ms more
                print 'rate NEST: ',self.get_firingrates()
                self.freq=[]
                self.psd=[]
                self.freq_log=[]
                self.psd_log=[]

                for i in range(self.K):
                   x=self.sim_A[NFFT:,i]
                   L=len(x)
                   x=x[:(L/NFFT)*NFFT].reshape((-1,NFFT))
                   ntrials=x.shape[0]
                   xF=fft(x)
                   S=np.sum(np.real(xF*xF.conjugate()),axis=0)*dt_sample/(NFFT-1)/ntrials
                   psd=S[1:NFFT/2]
                   freq=df*np.arange(NFFT/2-1)+df
                   f_log,psd_log = self.__rebin_log__(freq,psd,dpoints)
                   self.psd.append(psd)
                   self.freq.append(freq)
                   self.freq_log.append(f_log)
                   self.psd_log.append(psd_log)

                   self.psd=np.array(self.psd).T
                   self.freq=np.array(self.freq[0])



    def __get_rate_cv__(self, isih, dt):
        n=len(isih[:,0])
        npop=len(isih[0])
        t=(np.arange(n)+0.5)*dt
        m1=np.zeros(npop)
        m2=np.zeros(npop)
        v=np.zeros(npop)
        for i in range(npop):
            m1[i]=sum(t*isih[:,i])*dt
            m2[i]=sum(t*t*isih[:,i])*dt
            v[i]=m2[i]-m1[i]**2
        return (1./m1,np.sqrt(v)/m1)

    def get_isistat(self, tmax=2., Nbin=200, Nspikes=10000):
        print ''
        if self.__sim_mode__==None:
           print 'get_isistat(): No network has been built. Call build_network... first!'
        elif self.__sim_mode__=='pop_tilo':
           print '+++ get_isistat(): sim_mode must be netw_tilo or netw_tilo_record_voltage +++'
        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
           print '+++ GET ISIH +++'

        self.Nspikes=Nspikes
        self.dt_isi=tmax/Nbin

        fname=self.__isi_name__()
        print fname
        if os.path.exists(fname):
            print ''
            print 'LOAD EXISTING ISI DATA'
            X=np.loadtxt(fname)
            self.T_isi=X[:,0]
            self.isih=X[:,1:]
            dt=self.T_isi[1]-self.T_isi[0]
            self.rate,self.cv = self.__get_rate_cv__(self.isih, dt)
        else:
            if (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
                self.mp.get_isih_neuron(Nbin, self.dt_isi, Nspikes)
                self.T_isi=(np.arange(Nbin)+0.5) * self.dt_isi
                self.isih=self.mp.isih.T #each column corresponds to the psd of one population

                dt=self.T_isi[1]-self.T_isi[0]
                self.rate,self.cv = self.__get_rate_cv__(self.isih, dt)



    def get_firingrates(self, Tinit=0.1):
        nstart=int(Tinit/self.dt_rec)
        if (self.__sim_mode__=='populations' or self.__sim_mode__=='pop_tilo'):
           return np.mean(self.sim_a[nstart:],axis=0)
        else:
           return np.mean(self.sim_A[nstart:],axis=0)
    
    
    def plot_sim(self, title='',legend=None,t0=0):
        dt=self.sim_t[1]-self.sim_t[0]
        i0=int(t0/dt)
        pylab.figure(30)
        if (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
            pylab.plot( self.sim_t[i0:], self.sim_A[i0:])
        else:
            pylab.plot( self.sim_t[i0:], self.sim_a[i0:])
        pylab.show()



    def xm_sim(self, param='sim.par',t0=0):
        dt=self.sim_t[1]-self.sim_t[0]
        i0=int(t0/dt)
        try: 
            self.xmtrajec.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
        except: 
            self.xmtrajec=gracePlot(figsize=(720,540))
            self.xmtrajec.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)

        self.xmtrajec.focus(0,0)
        self.xmtrajec.plot( self.sim_t[i0:], self.sim_a[i0:])
        print self.sim_t[i0:]
        self.xmtrajec.grace('getp "%s"'%(param,))
        self.xmtrajec.grace('redraw')

        
    def plot_psd(self, title='',axis_scaling='loglog'):
        pylab.figure(10)
        if (axis_scaling=='loglog'):
            pylab.loglog( self.freq, self.psd)
        elif (axis_scaling=='semilogx'):
            pylab.semilogx( self.freq, self.psd)
        elif (axis_scaling=='semilogy'):
            pylab.semilogy( self.freq, self.psd)
        else:
            pylab.plot( self.freq, self.psd)
        pylab.xlabel('frequency [Hz]')
        pylab.ylabel('psd [Hz]')
        pylab.title(title)
        pylab.show()

    def xm_psd(self, param='psd.par'):
        try: 
            self.xmpsd.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
        except: 
            self.xmpsd=gracePlot(figsize=(720,540))
            self.xmpsd.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)

        self.xmpsd.focus(0,0)
        self.xmpsd.plot( self.freq, self.psd)
        self.xmpsd.grace('getp "%s"'%(param,))
        self.xmpsd.grace('redraw')


    def plot_voltage(self,k=0,offset=0):
        """
        plot voltage traces for population k 
        (1st population has index k=0)
        """
        if (self.Nrecord[k]>0):
            pylab.figure(20+k)
            Nbin=len(self.voltage[k])
            t=self.dt_rec*np.arange(Nbin)
            offset_matrix=np.outer(np.ones(Nbin),np.arange(self.Nrecord[k])) * offset
            pylab.plot(t,self.voltage[k]+offset_matrix)
            pylab.show()
        else:
            print 'Nrecord must be at least 1 to plot voltage!'

    def xm_voltage(self,k=0,offset=0, param='voltage.par'):
        """
        plot voltage traces for population k 
        (1st population has index k=0)
        """
        try: 
            self.xmvolt.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)
        except: 
            self.xmvolt=gracePlot(figsize=(720,540))
            self.xmvolt.multi(1,1,hgap=0.3,vgap=0.3,offset=0.15)

        self.xmvolt.focus(0,0)

        if (self.Nrecord[k]>0):
            Nbin=len(self.voltage[k])
            t=self.dt_rec*np.arange(Nbin)
            offset_matrix=np.outer(np.ones(Nbin),np.arange(self.Nrecord[k])) * offset
            self.xmvolt.plot(t,self.voltage[k]+offset_matrix)
            self.xmvolt.grace('getp "%s"'%(param,))
            self.xmvolt.grace('redraw')            
        else:
            print 'Nrecord must be at least 1 to plot voltage!'



    def __get_parameter_string__(self):


           
        if self.K>1:
           # if population mode take parameters of equivalent fully-connected network in file name
           if (self.__sim_mode__ == 'pop_tilo'):
              J1=self.J_syn[0][0] * self.pconn[0][0]
              J2=self.J_syn[0][1] * self.pconn[0][1]
              p1=1.
              p2=1.
           else:
              J1=self.J_syn[0][0]
              J2=self.J_syn[0][1]
              p1=self.pconn[0][0]
              p2=self.pconn[0][1]

           s='_mode%d_Npop%d_mu%g_du%g_vth%g_vr%g_c%g_J1_%g_J2_%g_p1_%g_p2_%g_taus1_%g_taus2_%g_taum%g_N1_%d_N2_%d_delay%g_tref%g_Na%d_Ja%g_taua%g_sigma%g'\
               %(self.mode,self.K,self.mu[0],self.delta_u[0],\
                    self.V_th[0], self.V_reset[0], self.rho_0[0], \
                    J1, J2, p1, p2, \
                    self.taus1[0][0], self.taus1[0][1], \
                    self.tau_m[0], self.N[0], self.N[1],\
                    self.delay[0][0],self.t_ref[0],len(self.J_a[0]),self.J_a[0][0],self.tau_sfa[0][0],self.sigma[0])
        else:
           # if population mode take parameters of equivalent fully-connected network in file name
           if (self.__sim_mode__ == 'pop_tilo'):
              J1=self.J_syn[0][0] * self.pconn[0][0]
              p1=1.
           else:
              J1=self.J_syn[0][0]
              p1=self.pconn[0][0]

           N_theta=len(self.J_a[0])
           if (N_theta>1):
              s='_mode%d_Npop%d_mu%g_du%g_vth%g_vr%g_c%g_J%g_p%g_taus1_%g_taum%g_N1_%d_delay%g_tref%g_Na%d_Ja1_%g_Ja2_%g_taua1_%g_taua2_%g_sigma%g'\
                 %(self.mode,self.K,self.mu[0],self.delta_u[0],\
                    self.V_th[0], self.V_reset[0], self.rho_0[0], \
                      J1, p1, \
                   self.taus1[0][0], \
                   self.tau_m[0],self.N[0], \
                   self.delay[0][0],self.t_ref[0],N_theta,self.J_a[0][0],self.J_a[0][1],self.tau_sfa[0][0],self.tau_sfa[0][1],self.sigma[0])

           else:
              s='_mode%d_Npop%d_mu%g_du%g_vth%g_vr%g_c%g_J%g_p%g_taus1_%g_taum%g_N1_%d_delay%g_tref%g_Na%d_Ja%g_taua%g_sigma%g'\
                 %(self.mode,self.K,self.mu[0],self.delta_u[0],\
                    self.V_th[0], self.V_reset[0], self.rho_0[0], \
                      J1, p1, \
                   self.taus1[0][0], \
                   self.tau_m[0],self.N[0], \
                   self.delay[0][0],self.t_ref[0],N_theta,self.J_a[0][0],self.tau_sfa[0][0],self.sigma[0])
        return s


    def __psd_name__(self):
        psd_str='_Ntrials%d'%(self.Ntrials,)
        str2='_dt%g_dtbin%g_df%g.dat'%(self.dt,int(self.dt_rec/self.dt)*self.dt,self.df)
        if self.__sim_mode__==None:
            print 'No network has been built. Call build_network... first!'
        elif self.__sim_mode__=='pop_tilo':
            return 'data/psd_pop' + self.__get_parameter_string__() + psd_str + str2
        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
            return 'data/psd_netw'+ self.__get_parameter_string__() + psd_str + str2
        elif (self.__sim_mode__=='neurons'):
            return 'data/psd_nestneur'+ self.__get_parameter_string__() + psd_str + str2
        elif (self.__sim_mode__=='populations'):
            return 'data/psd_nestpop'+ self.__get_parameter_string__() + psd_str + str2


    def __isi_name__(self):
        isi_str='_Nspikes%d'%(self.Nspikes,)
        str2='_dt%g_dtbin%g.dat'%(self.dt,int(self.dt_isi/self.dt)*self.dt)
        if self.__sim_mode__==None:
            print 'No network has been built. Call build_network... first!'
        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
            return 'data/isih_netw'+ self.__get_parameter_string__() + isi_str + str2


    def __trajec_name__(self):
       
        if self.step is not None:
           #use maximal step size in file name
           m=np.argmax(self.step)
           indx=np.unravel_index(m,self.step.shape)
           trajec_str='_step%g_tstep%g_T%g'%(self.step[indx],self.tstep[indx], self.sim_T)
        else:
           trajec_str='_T%g'%(self.sim_T,)
        str2='_dt%g_dtbin%g.npz'%(self.dt,int(self.dt_rec/self.dt)*self.dt)
        if self.__sim_mode__==None:
            print 'No network has been built. Call build_network... first!'
        elif (self.__sim_mode__=='netw_tilo') or (self.__sim_mode__=='netw_tilo_record_voltage'):
            return 'data/trajec_netw'+ self.__get_parameter_string__() + trajec_str + '_seed%d'%(self.seed,) + str2
        elif self.__sim_mode__=='pop_tilo':
            return 'data/trajec_pop'+ self.__get_parameter_string__() + trajec_str + '_seed%d'%(self.seed,) + str2
        elif self.__sim_mode__=='neurons':
            return 'data/trj_nrn'+ self.__get_parameter_string__() + trajec_str + str2
        elif self.__sim_mode__=='populations':
            return 'data/trj_pop'+ self.__get_parameter_string__() + trajec_str + str2
        else:
            assert(False)


    def save_psd(self):
        fname=self.__psd_name__()
        if os.path.exists(fname):
            print 'file already exists. PSD not saved again.'
        else:
            if not os.path.exists('data'):
                os.makedirs('data')
            np.savetxt(fname,np.c_[self.freq,self.psd],fmt='%g')
            print 'saved file ',fname

    def clean_psd(self):
        fname=self.__psd_name__()
        if os.path.exists(fname):
            os.remove(fname)  

    def clean_isih(self):
        fname=self.__psd_name__()
        if os.path.exists(fname):
            os.remove(fname)  

    def save_trajec(self):
        fname=self.__trajec_name__()
        if os.path.exists(fname):
            print 'file already exists. Trajectories not saved again.'
        else:
            if not os.path.exists('data'):
                os.makedirs('data')
            if self.__sim_mode__=='netw_tilo_record_voltage':
                np.savez(fname,t=self.sim_t,A=self.sim_A, a=self.sim_a,V=self.voltage,theta=self.threshold)
            else:
                np.savez(fname,t=self.sim_t,A=self.sim_A, a=self.sim_a)

    def clean_trajec(self, fname=None):
        if fname==None:
           fname=self.__trajec_name__()
        if os.path.exists(fname):
            os.remove(fname)    

    def clean_all(self):
        self.clean_psd()
        self.clean_trajec()
        self.clean_isih()



    def save_isih(self):
        fname=self.__isi_name__()
        if os.path.exists(fname):
            print 'file already exists. ISIH not saved again.'
        else:
            if not os.path.exists('data'):
                os.makedirs('data')

            np.savetxt(fname,np.c_[self.T_isi,self.isih],fmt='%g')
            print 'saved file ',fname

    def clean_isih(self):
        fname=self.__isi_name__()
        if os.path.exists(fname):
            os.remove(fname)  

            


    def get_singleneuron_rate_theory(self,i):
        """
        yields firing rate for a neuron in population i given constant input potential h
        no adaptation yet
        """
        tmax=0.
        dt=0.0001
        S_end=1
        while (S_end>0.001 and tmax<10.):
            tmax=tmax+1  #max ISI, in sec
            t=np.arange(0,tmax,dt)
            K=len(t)
            S=np.ones(K)
            if self.mode<10:
                eta=self.V_reset[i]*np.exp(-t/self.tau_m[i])
                rho=self.rho_0[i]*np.exp((self.mu[i]-eta-self.V_th[i])/self.delta_u[i])*(t>self.t_ref[i])
            else:
                v=self.mu[i]+(self.V_reset[i]-self.mu[i]) * np.exp(-(t-self.t_ref[i])/self.tau_m[i])
                rho=self.rho_0[i]*np.exp((v-self.V_th[i])/self.delta_u[i])*(t>self.t_ref[i])
            S=np.exp(-np.cumsum(rho)*dt)
            S_end=S[-1]
        return 1./(np.sum(S)*dt)

    def get_spike_afterpotential(self,k=0,tmax=0.5):
        t=linspace(0,tmax,200)
        v=self.mu+(self.V_reset-self.mu) * exp(-(t-self.t_ref)/self.tau_m)*(t>=self.t_ref)+ (t<self.t_ref)*self.V_reset


    def get_threshold_kernel(self,k=0,tmax=0.5,dt=0.001):
#       t=linspace(0,tmax,201)
       t=np.arange(0,tmax+0.5*dt,dt)
       n=len(t)
       theta=np.zeros(n)
       for i in range(len(self.J_a[k])):
          Ja=self.J_a[k][i]
          taua=self.tau_sfa[k][i]
          theta += Ja / taua * np.exp(-t / taua)
       return (t,theta)

            
def get_dataframe(nest_id):
    assert(type(nest_id)==int)
    return pandas.DataFrame( nest.GetStatus( [nest_id], keys=['events'] )[0][0])



