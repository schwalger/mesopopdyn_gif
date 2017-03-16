#! /usr/bin/env python

from pylab import *
import sys



def array_to_str(x):
    if type(x) is np.ndarray:
        return str(x.tolist()).replace('[','{').replace(']','}')
    elif type(x) is list:
        return str(x).replace('[','{').replace(']','}')
    else:
        print 'neither array nor list'


def format_fbkernel_params(Ja,taua):
    J_theta=[]
    tau_theta=[]
    for i in range(len(Ja)):
        L=Ja[i]
        TAU=taua[i]
        if L==[]:
            J_theta.append([])
            tau_theta.append([])
        else:
            Jlist=[]
            taulist=[]
            for j in range(len(L)):
                J=L[j]
                if J!=0.:
                    Jlist.append(J)
                    taulist.append(TAU[j])
            J_theta.append(Jlist)
            tau_theta.append(taulist)

        N_theta=[len(i) for i in J_theta]

    return (J_theta,tau_theta,N_theta)


def generate_paramfile(p,fname='param.h'):

    f = open(fname, 'w')
    f.write('#define NPOP %d\n'%(p.K,))
    
    f.write('static int N[NPOP] = {' + ', '.join(map(str,p.N)) + '};\n')
    f.write('static double mu[NPOP] = {' + ', '.join(map(str,p.mu)) + '};\n')
    f.write('static double DeltaV[NPOP] = {' + ', '.join(map(str,p.delta_u)) + '};\n')
    f.write('static double c[NPOP] = {' + ', '.join(map(str,p.rho_0)) + '};\n')
    f.write('static double tref[NPOP] = {' + ', '.join(map(str,p.t_ref)) + '};\n')
    f.write('static double delay[NPOP] = {' + ', '.join(map(str,p.delay[0])) + '};\n')
    f.write('static double vth[NPOP] = {' + ', '.join(map(str,p.V_th)) + '};\n')
    f.write('static double vreset[NPOP] = {' + ', '.join(map(str,p.V_reset)) + '};\n')
    f.write('static double taum[NPOP] = {' + ', '.join(map(str,p.tau_m)) + '};\n')
    f.write('static double sigma[NPOP] = {' + ', '.join(map(str,p.sigma)) + '};\n')
    f.write('static double taus1[NPOP][NPOP] = ' + array_to_str(p.taus1) + ';\n')
    f.write('static double taus2[NPOP][NPOP] = ' + array_to_str(p.taus2) + ';\n')
    f.write('static double taur1[NPOP][NPOP] = ' + array_to_str(p.taur1) + ';\n')
    f.write('static double taur2[NPOP][NPOP] = ' + array_to_str(p.taur2) + ';\n')
    f.write('static double a1[NPOP][NPOP] = ' + array_to_str(p.a1) + ';\n')
    f.write('static double a2[NPOP][NPOP] = ' + array_to_str(p.a2) + ';\n')
    f.write('static double J[NPOP][NPOP] = ' + array_to_str(p.J_syn) + ';\n')
    f.write('static double p_conn[NPOP][NPOP] = ' + array_to_str(p.pconn) + ';\n')
    
    
    J_theta,tau_theta,N_theta = format_fbkernel_params(p.J_a,p.tau_sfa)
    f.write('static int N_theta[NPOP]=' + array_to_str(N_theta) + ';\n')
    f.write('static double J_ref[NPOP]=' + array_to_str(np.zeros(p.K)) + ';\n\n\n')
    
    
    f.write('void fill_fbkernel(double *J_theta[], double *tau_theta[]){\n')
    for k in range(p.K):
        f.write('static double J_theta%d[%d]='%(k,N_theta[k]) + array_to_str(J_theta[k]) + ';\n')
        f.write('J_theta[%d] = J_theta%d;\n'%(k,k))
        f.write('static double tau_theta%d[%d]='%(k,N_theta[k]) + array_to_str(tau_theta[k]) + ';\n')
        f.write('tau_theta[%d] = tau_theta%d;\n'%(k,k))
    f.write('}\n\n\n')
        
    f.write('char paramstr[]="'+p.__get_parameter_string__() + '";\n')
        
        
    f.close()

