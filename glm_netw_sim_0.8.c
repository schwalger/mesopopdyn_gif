//changes to version glm_netw_sim_0.6.c 
//separate seed for quenched randomness of the connectivity matrix
// this allows to run several trials of trajectories for the same network topology






#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "util.c"
#include <string.h>

static double DT;
static double DTBIN;
static double VSPIKE;  //peak voltage at spike in mV, used to mark spikes in voltage trace

#ifndef GLM
#define GLM 0
#endif

#ifndef GLIF
#define GLIF 10
#endif

#define PFIRE_MAX 0.99999
#define PFIRE_MIN 0.00001



//data type for single neuron variables
struct Neuron {
  double h;
  double eta; //refractoriness
  double *Xsyn1; //incoming synaptic currents (filtered with fast decay time, e.g. AMPA, GABA)
  double *Xsyn2; //incoming synaptic currents (filtered with slow decay time, e.g. NMDA)
  double *Isyn1; //total incoming synaptic current (sum of Xsyn1+Xsyn2 filtered with short rise-time)
  double *Isyn2; //total incoming synaptic current (sum of Xsyn1+Xsyn2 filtered with short rise-time)
  double *theta; //dynamic threshold
  double tlast; //time since last spike
  char *axon;
  int offset; //current offset of axon
  int *target_start; //
  int *target_end; //
};



//data type for population parameters
struct Population{
  double tref;  //absolute refractory period
  double taum;
  double *taus1; //incoming synaptic time constants, decay time of 1st filter
  double *taus2; //incoming synaptic time constants, decay time of 2nd filter
  double *taur1; //incoming synaptic time constants, rise time of 1st filter
  double *taur2; //incoming synaptic time constants, rise time of 2nd filter
  double *a1; //relative weight of fast synaptic current 
  double *a2; //relative weight of slow synaptic current 
  double mu;
  double c;
  double deltaV;
  double delay;
  double vth;          //baseline threshold
  double J_ref;        // strength of refractory kernel in mV*ms (kernel assumed to be exponential with time constant taum and amplitude J_ref/taum)
  double h_reset;      //reset potential for h in glif mode
  double *theta_jump;      //strength of threshold kernel in mV (kernel assumed to be sum of exponentials with time constants tau_theta and jump amplitude J_theta/tau_theta)
  double *tau_theta;   //time constant of spike-triggered threshold kernel
  int N_theta;         //number of exponentials for threshold kernel
  int N;               //number of neurons in population
  double *J;           //incoming synaptic weights in mV
  double *Iext;        //external input; if no external input, initialize with Iext=NULL
  double *p_conn;      //connection probability, p_conn * N = number of incoming connections per presynaptic population (in-degree)

  //internal parameters
  int Npop;            //number of (incoming) populations
  double *w1;          // effective weight
  double *w2;
  double *Es1;          //exp(-dt/taus1)
  double *Es2;          //exp(-dt/taus2)
  double *Er1;          //exp(-dt/taus1)
  double *Er2;          //exp(-dt/taus2)
  double *E_theta;     //exp(-dt/tau_theta)
  double E;            //exp(-dt/taum)
  double g;        //for white noise RIext=sqrt(2/taum)sigma\xi(t), <xi>=0
  int Nisi;  // number of bins for ISIH, if 0 do not measure isih
  int spikecount;
  int *isih;
  double dV;     //V=u-theta; dV=deltaV/100 is resolution of voltage in lookup table for Pfire
  double Vmin;   //minimum V in lookup table for Pfire
  double Vmax;
  double *Pfire;  //lookup table
};




void setup_simulation_method(void (* update_neuron_state[])(struct Neuron *,struct Population *,int, int, gsl_rng *),  void (* receive_spike[])(struct Neuron *,struct Population *,int), int Npop, char psc_type[Npop][30]);
void print_pop_parameters(struct Population p[], int Npop);
unsigned long int random_seed();



void init_population(struct Population p[], int Npop,double tref[], double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[], int mode, double dt)
  
{

  int k,l;
  for (k=0;k<Npop;k++)
    {
      p[k].tref=tref[k];
      p[k].taum=taum[k];
      p[k].mu=mu[k];
      p[k].c=c[k];
      p[k].deltaV=deltaV[k];
      p[k].delay=delay[k];
      p[k].vth=vth[k];
      if (mode>=GLIF) 
	p[k].h_reset=vreset[k];
      else p[k].h_reset=1000.; //mV, set to impossible value to indicate GLM mode 
      p[k].N=N[k];
      p[k].J=J[k];
      p[k].p_conn=p_conn[k];
      p[k].Iext=signal[k];
      p[k].N_theta=N_theta[k];
      p[k].J_ref=J_ref[k];//should be zero in GLIF mode
      p[k].theta_jump=dvector(N_theta[k]);
      for (l=0;l<N_theta[k];l++)
	p[k].theta_jump[l] = J_theta[k][l] / tau_theta[k][l];
      p[k].tau_theta=tau_theta[k];
      p[k].taus1=taus1[k];
      p[k].taus2=taus2[k];
      p[k].taur1=taur1[k];
      p[k].taur2=taur2[k];
      p[k].a1=a1[k];
      p[k].a2=a2[k];
      p[k].g=sqrt(1-exp(-2*dt/p[k].taum)) * sigma[k];
      p[k].Nisi=0;  //by default, do not measure isih

      //create lookup table for firing probability
      p[k].dV=p[k].deltaV/100;
      p[k].Vmin = p[k].deltaV * log(-log(1-PFIRE_MIN)/dt / p[k].c);
      p[k].Vmax = p[k].deltaV * log(-log(1-PFIRE_MAX)/dt / p[k].c);
      int L=(p[k].Vmax - p[k].Vmin)/p[k].dV;
      //      printf("Lookup tbl (popul %d): Use L=%d, Vmin=%g Vmax=%g dV=%g\n",k+1,L,p[k].Vmin,p[k].Vmax,p[k].dV);
      p[k].Pfire=dvector(L);
      for (l=0;l<L;l++)
	{
	  double V=p[k].Vmin + l*p[k].dV;
	  p[k].Pfire[l]= -expm1(-p[k].c * exp(V/p[k].deltaV) * dt);
	}

    }

}

void free_population(int Npop,struct Population p[])
{
  int k;
  for (k=0;k<Npop;k++)
    {
      free_dvector(p[k].theta_jump);
      free_dvector(p[k].Pfire);
    }
}


struct Neuron **initialize_neurons(int Npop,struct Population p[])
{
  int i,j,k;
  struct Neuron **neurons;
  neurons=(struct Neuron **) malloc((size_t)(Npop*sizeof(struct Neuron*)));

  for (k=0;k<Npop;k++)
    {
      int ndelay=p[k].delay/DT;
      neurons[k]=malloc(sizeof(struct Neuron)*p[k].N);

      for (i=0;i<p[k].N;i++)
	{
	  //	  neurons[k][i].h=p[k].vth-drand48()*20;
	  neurons[k][i].h=0.;
	  neurons[k][i].eta=0;

	  neurons[k][i].Xsyn1=dvector(Npop);
	  for (j=0;j<Npop;j++) neurons[k][i].Xsyn1[j]=0;
	  neurons[k][i].Xsyn2=dvector(Npop);
	  for (j=0;j<Npop;j++) neurons[k][i].Xsyn2[j]=0;
	  neurons[k][i].Isyn1=dvector(Npop);
	  for (j=0;j<Npop;j++) neurons[k][i].Isyn1[j]=0;
	  neurons[k][i].Isyn2=dvector(Npop);
	  for (j=0;j<Npop;j++) neurons[k][i].Isyn2[j]=0;

	  neurons[k][i].theta=dvector(p[k].N_theta);
	  for (j=0;j<p[k].N_theta;j++) neurons[k][i].theta[j]=0;

	  neurons[k][i].tlast=0;
	  neurons[k][i].axon=(char*)malloc(sizeof(char)*ndelay);
	  for (j=0;j<ndelay;j++) neurons[k][i].axon[j]=0;
	  neurons[k][i].offset=0;
	  neurons[k][i].axon[0]=1;
	  neurons[k][i].target_start=(int*)malloc(sizeof(int)*Npop);
	  neurons[k][i].target_end=(int*)malloc(sizeof(int)*Npop);
	}
    }

  return neurons;
}


void free_neurons(struct Neuron **neurons,int Npop,struct Population p[])
{
  int i,k;

  for (k=0;k<Npop;k++)
    {
      for (i=0;i<p[k].N;i++)
	{
	  free_dvector(neurons[k][i].Xsyn1);
	  free_dvector(neurons[k][i].Xsyn2);
	  free_dvector(neurons[k][i].Isyn1);
	  free_dvector(neurons[k][i].Isyn2);
	  free_dvector(neurons[k][i].theta);
	  free(neurons[k][i].axon);
	  free(neurons[k][i].target_start);
	  free(neurons[k][i].target_end);
	}

      free(neurons[k]);
    }

}


unsigned short ***allocate_target_tensor(int Npop)
{
  int i;
  unsigned short ***targets;
  targets=(unsigned short ***) malloc((size_t)((Npop)*sizeof(unsigned short**)));
  targets[0]=(unsigned short **) malloc((size_t)((Npop*Npop)*sizeof(unsigned short*)));
  for(i=1;i<Npop;i++) targets[i]=targets[i-1]+Npop;
  return targets;
}



void free_target_tensor(unsigned short ***targets,int Npop)
{
  int i,j;
  for (i=0;i<Npop;i++)
    for (j=0;j<Npop;j++)
      free(targets[i][j]);
  free(targets[0]);
  free(targets);
}


double get_mean(double *x,int n)
{
  int i;
  double m=0;
  for (i=0;i<n;i++) m+=x[i];
  return m/n;
}

void construct_random_connectivity(unsigned short ***targets,struct Population p[],struct Neuron *neuron[],int Npop,gsl_rng *rng)
{
  //  printf("Constructing random matrix ...\n");

  int i,j,k,l;

  for (j=0;j<Npop;j++)
    {
      int *indx;
      indx=ivector(p[j].N);
      for (k=0;k<p[j].N;k++) indx[k]=k;

      for (i=0;i<Npop;i++)
	{
	  printf("Connecting population %d with %d\r",j+1,i+1);
	  fflush(stdout);

	  if (p[i].p_conn[j]==1) targets[i][j]=NULL;
	  else
	    { 
	      int C=p[i].p_conn[j] * p[j].N;
	      targets[i][j]=(unsigned short *)malloc((size_t)(C * p[i].N*sizeof(unsigned short)));
	      
	      int **afferents;
	      afferents=imatrix(p[i].N, C);

	      //Choose presynaptic neuron set	      
	      for (k=0;k<p[i].N;k++)
		gsl_ran_choose(rng,afferents[k],C,indx,p[j].N,sizeof(int));
	      
	      //Create postsynaptic neuron set	      
	      int m,*start,n=0;
	      start=ivector(p[i].N);
	      for (k=0;k<p[i].N;k++) start[k]=0;
	      
	      for (m=0;m<p[j].N;m++)
		{
		  
		  neuron[j][m].target_start[i]=n;
		  
		  for (k=0;k<p[i].N;k++)
		    for (l=start[k];l<C;l++)
		      if (m<=afferents[k][l])
			{ 
			  if (afferents[k][l]==m) 
			    {
			      targets[i][j][n]=k;
			      n++;
			      start[k]=l+1;
			    }
			  break;
			}
		      
				  
		  neuron[j][m].target_end[i]= n - 1;
		}
	      
	      free_imatrix(afferents);

	    }
	}
    }
}

void init_internal_population_variables(struct Population p[], int Npop, char psc_type[Npop][30])
{

  int j,k;
  for (j=0;j<Npop;j++) 
    {
            p[j].Npop=Npop;
      p[j].E=exp(-DT/p[j].taum);
      p[j].Es1=dvector(Npop);
      p[j].Es2=dvector(Npop);
      p[j].Er1=dvector(Npop);
      p[j].Er2=dvector(Npop);
      p[j].w1=dvector(Npop);
      p[j].w2=dvector(Npop);
      for (k=0; k<Npop; k++)
	{
	  if (p[j].taus1[k]>0)
	    {
	      p[j].w1[k]=p[j].J[k] * p[j].taum / p[j].taus1[k] * p[j].a1[k] / (p[j].a1[k]+p[j].a2[k]);
	      p[j].Es1[k] = exp(-DT / p[j].taus1[k]);
	      if (p[j].taur1[k]>0) p[j].Er1[k] = exp(-DT / p[j].taur1[k]);
	      else p[j].Er1[k] = 0;
	    }
	  else if (p[j].taur1[k]>0)
	    {
	      p[j].w1[k]=p[j].J[k] * p[j].taum / p[j].taur1[k] * p[j].a1[k] / (p[j].a1[k]+p[j].a2[k]);
	      p[j].Es1[k] = 0;
	      p[j].Er1[k] = exp(-DT / p[j].taur1[k]);
	    }
	  else 
	    {	  
	      //	      printf("HAllo k=%d a1[k]=%g a2[k]=%g J[k]=%g\n",k,p[j].a1[k],p[j].a2[k],p[j].J[k]);
	      p[j].w1[k]=p[j].J[k] * p[j].a1[k] / (p[j].a1[k]+p[j].a2[k]);
	      p[j].Es1[k] = 0;
	      p[j].Er1[k] = 0;
	    }

	  //the same for slow (second) filter
	  if (p[j].taus2[k]>0)
	    {
	      p[j].w2[k]=p[j].J[k] * p[j].taum / p[j].taus2[k] * p[j].a2[k] / (p[j].a1[k]+p[j].a2[k]);
	      p[j].Es2[k] = exp(-DT / p[j].taus2[k]);
	      if (p[j].taur2[k]>0) p[j].Er2[k] = exp(-DT / p[j].taur2[k]);
	      else p[j].Er2[k] = 0;
	    }
	  else if (p[j].taur2[k]>0)
	    {
	      p[j].w2[k]=p[j].J[k] * p[j].taum / p[j].taur2[k] * p[j].a2[k] / (p[j].a1[k]+p[j].a2[k]);
	      p[j].Es2[k] = 0;
	      p[j].Er2[k] = exp(-DT / p[j].taur2[k]);
	    }
	  else 
	    {	  
	      p[j].w2[k]=p[j].J[k] * p[j].a2[k] / (p[j].a1[k]+p[j].a2[k]);
	      p[j].Es2[k] = 0;
	      p[j].Er2[k] = 0;
	    }
	  
	} 
      if (p[j].N_theta>0)
	{
	  p[j].E_theta=dvector(p[j].N_theta);
	  for (k=0; k< p[j].N_theta; k++) p[j].E_theta[k]=exp(-DT/p[j].tau_theta[k]);
	}
      else p[j].E_theta=NULL;
    
      //determine if psc shape is simple exponential for all incoming currents to pop j
      strcpy(psc_type[j], "generic");
      int all_zero=1;
      for (k=0;k<Npop;k++)
	if ((p[j].taur1[k]>0) || (p[j].taur2[k]>0)) all_zero=0;
      if (all_zero==1) strcpy(psc_type[j], "exponential");

      //check if all synaptic time constants are zero (rise and decay times)
      all_zero=1;
      for (k=0;k<Npop;k++)
	if ((p[j].taur1[k]>0) || (p[j].taur2[k]>0) || (p[j].taus1[k]>0) || (p[j].taus2[k]>0)) all_zero=0;
      if (all_zero==1) strcpy(psc_type[j], "delta");
    }

}


void free_internal_population_variables(struct Population p[], int Npop)
{
  int j;
  for (j=0;j<Npop;j++)
    {
      free_dvector(p[j].Es1);
      free_dvector(p[j].Es2);
      free_dvector(p[j].Er1);
      free_dvector(p[j].Er2);
      free_dvector(p[j].w1);
      free_dvector(p[j].w2);
      if (p[j].N_theta>0) free_dvector(p[j].E_theta);
    }

}




void transfer_spikes(struct Neuron *neuron[],struct Population p[],unsigned short ***targets,int Npop, void (* receive_spike[])(struct Neuron *, struct Population *,int))
{
  int i,j,k,l,n;
  unsigned short *target_neurons;
  for (k=0;k<Npop;k++)
    for (i=0;i<p[k].N;i++)
      {
	//	printf("i=%d\n",i);
	if (neuron[k][i].axon[neuron[k][i].offset]==1)
	  {
	    for (l=0;l<Npop;l++)
	      {
		if (p[l].p_conn[k]<=0) continue;
		else 
		  {
		    if (p[l].p_conn[k]>=1.)  //full connectivity
		      for (j=0;j<p[l].N;j++) receive_spike[l](neuron[l]+j,p+l,k);

		    else               //random connectivity
		      {
			target_neurons=targets[l][k];
			for (n=neuron[k][i].target_start[l];n<=neuron[k][i].target_end[l];n++) 
			  {
			    j=target_neurons[n]; 
			    receive_spike[l](neuron[l]+j,p+l,k);
			  }
		      }
		  }
	      }  
	    neuron[k][i].axon[neuron[k][i].offset]=0; // remove spike from axon
	  }
      }
}




double pspike(struct Neuron *n,struct Population *p)
{
  double vth,V;
  int i;
  if (n->tlast > p->tref) 
    {
      vth = p->vth;
      for (i=0;i< p->N_theta;i++) vth += n->theta[i];
      V=n->h + n->eta - vth;
      if (V < p->Vmax)
	{
	  if (V>=p->Vmin) return p->Pfire[(int)((V-p->Vmin)/p->dV)];
	  else return 0.;
	}
      else return 1.;
    }
  else return 0.;
}


void get_spikes_glm(struct Neuron *neurons,int *Abin,struct Population *p, int mode, gsl_rng *rng)
{
  int i,k,n=p->N,ndelay=p->delay/DT;
  for (i=0;i<n;i++)
    {
      if (gsl_rng_uniform(rng)<=pspike(neurons+i,p))
	{
	  neurons[i].axon[neurons[i].offset]=1; // send spike into axon

	  if (p->Nisi>0)
	    {
	      int indx=neurons[i].tlast/DTBIN;
	      if (indx<p->Nisi) p->isih[indx]++;
	      (p->spikecount)++;
	    }

	  neurons[i].tlast=0;

	  neurons[i].eta -= p->J_ref;

	  for (k=0; k < p->N_theta; k++) 
	    neurons[i].theta[k]+=p->theta_jump[k];
	  (*Abin)++;

	  //reset for glif model
	  if (mode==GLIF) neurons[i].h = p->h_reset;
	}
      neurons[i].offset=(neurons[i].offset+1)%ndelay;
      neurons[i].tlast+=DT;
    }
}




void receive_spike_pscgeneric(struct Neuron *neuron, struct Population *p, int k)
//neuron from population p receives a spike from some neuron in population k
{
  if (p->taus1[k] > 0) neuron->Xsyn1[k] += 1.;
  else if (p->taur1[k]>0) neuron->Isyn1[k] += 1.;
  else if (p->h_reset==1000) //GLM mode
    neuron->h += p->w1[k];
  else if (neuron->tlast > p->tref)
    neuron->h += p->w1[k];

  if (p->taus2[k] > 0) neuron->Xsyn2[k] += 1.;
  else if (p->taur2[k]>0) neuron->Isyn2[k] += 1.;
  else if (p->h_reset==1000) //GLM mode
    neuron->h += p->w2[k];
  else if (neuron->tlast > p->tref)
    neuron->h += p->w2[k];


}


void receive_spike_pscexponential(struct Neuron *neuron, struct Population *p, int k)
//neuron from population p receives a spike from some neuron in population k
//with exponential psc
{
  neuron->Xsyn1[k] += 1.;
  neuron->Xsyn2[k] += 1.;
}

void receive_spike_pscdelta(struct Neuron *neuron, struct Population *p, int k)
//neuron from population p receives a spike from some neuron in population k
// delta psc's
{
  if (p->h_reset==1000) //GLM mode
    neuron->h += p->w1[k];
  else if (neuron->tlast > p->tref)
    neuron->h += p->w1[k];

  if (p->h_reset==1000) //GLM mode
    neuron->h += p->w2[k];
  else if (neuron->tlast > p->tref)
    neuron->h += p->w2[k];
}










void update_neuron_state_pscgeneric(struct Neuron *neurons,struct Population *p,int k, int mode, gsl_rng *rng)
//generic update routine
{
  int i,j;
  double mu,input;

  if (p->Iext != NULL) 
    mu = p->mu + p->Iext[k]; 
  else mu=p->mu;

  double xi=gsl_ran_gaussian_ziggurat(rng, p->g);

  for (i=0; i < p->N; i++)
    {
      input=mu;
      for (j=0; j < p->Npop; j++)
	{
	  if (p->taur1[j] > 0)
	    {
	      input+=p->w1[j]*neurons[i].Isyn1[j];
	      if (p->taus1[j] > 0)
		{
		  neurons[i].Isyn1[j] = neurons[i].Xsyn1[j] + (neurons[i].Isyn1[j] - neurons[i].Xsyn1[j]) * p->Er1[j];
		  neurons[i].Xsyn1[j]*=p->Es1[j];  
		}
	      else 
		neurons[i].Isyn1[j]*= p->Er1[j];
	    }
	  else //zero-rise time
	    if (p->taus1[j] > 0)
	      {
		input+=p->w1[j]*neurons[i].Xsyn1[j];
		neurons[i].Xsyn1[j]*=p->Es1[j];  
	      }

	  //the same for second filter
	  if (p->taur2[j] > 0)
	    {
	      input+=p->w2[j]*neurons[i].Isyn2[j];
	      if (p->taus2[j] > 0)
		{
		  neurons[i].Isyn2[j] = neurons[i].Xsyn2[j] + (neurons[i].Isyn2[j] - neurons[i].Xsyn2[j]) * p->Er2[j];
		  neurons[i].Xsyn2[j]*=p->Es2[j];  
		}
	      else 
		neurons[i].Isyn2[j]*= p->Er2[j];
	    }
	  else //zero-rise time
	    if (p->taus2[j] > 0)
	      {
		input+=p->w2[j]*neurons[i].Xsyn2[j];
		neurons[i].Xsyn2[j]*=p->Es2[j];  
	      }
	}    

      if (mode==GLM)
	neurons[i].h=input+(neurons[i].h-input) * p->E + xi;
      else if (neurons[i].tlast > p->tref) //in GLIF mode only update h after refractory period      
	neurons[i].h=input+(neurons[i].h-input) * p->E + xi;

      neurons[i].eta *= p->E;

      for (j=0; j < p->N_theta; j++) 
	neurons[i].theta[j]*=p->E_theta[j];
    }
}


void update_neuron_state_pscexponential(struct Neuron *neurons,struct Population *p,int k, int mode, gsl_rng *rng)
//update routine for single exponential psc's (zero rise time)
{
  int i,j;
  double mu, input;

  if (p->Iext != NULL) 
    mu = p->mu + p->Iext[k]; 
  else mu = p->mu;

  double xi=gsl_ran_gaussian_ziggurat(rng,p->g);

  for (i=0; i < p->N; i++)
    {
      input=mu;
      for (j=0; j < p->Npop; j++)
	{
	  input+=p->w1[j]*neurons[i].Xsyn1[j];
	  neurons[i].Xsyn1[j]*=p->Es1[j];  
	}    

      if (mode==GLM)
	neurons[i].h=input+(neurons[i].h-input) * p->E + xi;
      else if (neurons[i].tlast > p->tref) //in GLIF mode only update h after refractory period      
	neurons[i].h=input+(neurons[i].h-input) * p->E + xi;

      neurons[i].eta *= p->E;

      for (j=0; j < p->N_theta; j++) 
	neurons[i].theta[j]*=p->E_theta[j];
    }
}


void update_neuron_state_pscdelta(struct Neuron *neurons,struct Population *p,int k, int mode, gsl_rng *rng)
//update routine for delta/instantaneous synapses (zero rise and decay time)
{
  int i,j;
  double mu;

  if (p->Iext != NULL) 
    mu = p->mu + p->Iext[k]; 
  else mu=p->mu;

  double xi=gsl_ran_gaussian_ziggurat(rng,p->g);

  for (i=0; i < p->N; i++)
    {
      if (mode==GLM)
	neurons[i].h=mu+(neurons[i].h-mu) * p->E + xi;
      else if (neurons[i].tlast > p->tref) //in GLIF mode only update h after refractory period      
	neurons[i].h=mu+(neurons[i].h-mu) * p->E + xi;

      neurons[i].eta *= p->E;

      for (j=0; j < p->N_theta; j++) 
	neurons[i].theta[j]*=p->E_theta[j];
    }
}






void simulate(int **Abin,int Tbin,struct Neuron **neurons,struct Population p[],unsigned short ***targets,int Npop,gsl_rng *rng,int mode, int dispprog)
//adds spikes to Abin[0..Tbin-1]
//if dispprog==1 display progress of simulation in percent
{
  int nbin=DTBIN/DT,n,k,A[Npop],j;
  for (j=0;j<Npop;j++) A[j]=0;
  char psc_type[Npop][30];

  init_internal_population_variables(p, Npop, psc_type);


  void (* update_neuron_state[Npop])(struct Neuron *,struct Population *,int, int, gsl_rng *);
  void (* receive_spike[Npop])(struct Neuron *, struct Population *, int);

  setup_simulation_method(update_neuron_state, receive_spike, Npop, psc_type);

  //step size for which to display progress
  int dispcount=(int)(Tbin/100);
  if (dispcount==0) dispcount=Tbin + 1;

  for (k=0;k<Tbin;k++)
    {
      for (n=0;n<nbin;n++)
	{
	  transfer_spikes(neurons,p,targets,Npop,receive_spike);
	  for (j=0;j<Npop;j++) update_neuron_state[j](neurons[j],p+j,k, mode,rng);
	  for (j=0;j<Npop;j++) get_spikes_glm(neurons[j],A+j,p+j,mode,rng);
	}

      for (j=0;j<Npop;j++) 
	{
	  Abin[j][k]=A[j];
	  A[j]=0;
	}

      if (dispprog)
	if ((k+1)%dispcount==0)
	  {
	    //	printf("k+1=%d dc=%d mod=%d perc=%d\r",k+1,dispcount, (k+1)%dispcount,(int)((k+1) * 100. / Tbin));
	    printf("%d%%  \r",(int)((k+1) * 100. / Tbin));
	    fflush(stdout);
	  }

    }

  free_internal_population_variables(p, Npop);
 
}



void get_trajectory_with_fullparameterlist(double **A,int Nbin,int Npop,double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[], double dt, double dtbin,int mode,int seed,int seed_quenched)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  DT=dt;
  DTBIN=dtbin;

  struct Population p[Npop];
  init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta, tau_theta, sigma, mode, DT);
  //  printf("seed=%d\n",seed);
  gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set (rng,(long)seed);

  gsl_rng *rng_quenched=gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set (rng_quenched,(long)seed_quenched);

  struct Neuron **neurons;
  neurons=initialize_neurons(Npop,p);

  int **Abin;
  Abin=imatrix(Npop,Nbin);
  int i,j;

  unsigned short ***targets;
  targets=allocate_target_tensor(Npop);

  construct_random_connectivity(targets,p,neurons,Npop,rng_quenched);

  clock_t start=clock();
  simulate(Abin,Nbin,neurons,p,targets,Npop,rng,mode,1);
  double sim_t=(double)(clock()-start)/CLOCKS_PER_SEC;
  printf("Execution time of microscopic dynamics: %g seconds, %g s per biosecond\n",sim_t, sim_t/Nbin/DTBIN);

  for (j=0;j<Npop;j++)
    for (i=0;i<Nbin;i++)
      A[j][i]=(double)(Abin[j][i])/dtbin/p[j].N;

  for (j=0;j<Npop;j++) printf("%g ",get_mean(A[j],Nbin));
  printf("\n");

  gsl_rng_free (rng);
  gsl_rng_free (rng_quenched);
  free_imatrix(Abin);
  free_target_tensor(targets,Npop);
  free_neurons(neurons,Npop,p);
  free_population(Npop,p);
}



void get_trajectory_srm_with_2D_arrays(int Nbin, double AA[][Nbin], int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double s[][Nbin], int N_theta[], double J_ref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode, int seed, int seed_quenched)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  //convert 2D array to double**
  double *AAA[Npop],*signal[Npop], *J_theta_ptr[Npop], *tau_theta_ptr[Npop];
  int i;

  for (i=0;i<Npop;i++)
    AAA[i]=AA[i];
  for (i=0;i<Npop;i++)
    signal[i]=s[i];


  int indx=0;
  for (i=0;i<Npop;i++)
    {
      /* printf("N=%d\n",N[i]); */
      /* printf("Ntheta=%d\n",N_theta[i]); */
      if (N_theta[i]>0)
  	{
  	  J_theta_ptr[i]=&(J_theta[indx]);
  	  tau_theta_ptr[i]=&(tau_theta[indx]);
  	  indx+=N_theta[i];
  	}
      else
  	{
  	  J_theta_ptr[i]=NULL;
  	  tau_theta_ptr[i]=NULL;
  	}
    }

  get_trajectory_with_fullparameterlist(AAA, Nbin, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta_ptr, tau_theta_ptr, sigma, dt, dtbin, mode, seed, seed_quenched);

}




void get_psd(double **SA,int Nbin,int Ntrials,struct Population p[],int Npop,double dtbin,int mode)
{
  gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2);
  unsigned long int seed;
  seed = random_seed();
  gsl_rng_set(rng,seed);
  //  gsl_rng_set (rng,(long)time(NULL));

  struct Neuron **neurons;
  neurons=initialize_neurons(Npop,p);

  int **Abin;
  double **A;
  Abin=imatrix(Npop,Nbin);
  A=dmatrix(Npop,Nbin);
  int i,j;

  unsigned short ***targets;
  targets=allocate_target_tensor(Npop);

  construct_random_connectivity(targets,p,neurons,Npop,rng);

  //warmup
  simulate(Abin,Nbin,neurons,p,targets,Npop,rng,mode,0);

  int n;
  double complex *AF[Npop];
  
  for (j=0;j<Npop;j++)
    {
      //      printf("Nbin=%d\n",Nbin/2);
      AF[j]=(double complex *)malloc(sizeof(double complex)*Nbin);
      for (i=0;i<Nbin/2;i++) SA[j][i]=0;
    }
  fftw_plan plan=fftw_plan_dft_r2c_1d(Nbin,A[0],AF[0],FFTW_MEASURE);


  for (n=0;n<Ntrials;n++)
    {
      simulate(Abin,Nbin,neurons,p,targets,Npop,rng,mode,0);

      for (j=0;j<Npop;j++)
	{
	  for (i=0;i<Nbin;i++) A[j][i]=(double)(Abin[j][i])/dtbin/p[j].N;
	  fftw_execute_dft_r2c(plan,A[j],AF[j]);
	  for (i=1;i<Nbin/2+1;i++) SA[j][i-1]+=creal(AF[j][i]*conj(AF[j][i]))*dtbin/Nbin;
	}

      //print trial information
      //      if (n%10==9)
	{
	  printf("trial %d ",n+1); 
          for (j=0;j<Npop;j++) printf("%g ",get_mean(A[j],Nbin));
	  printf("\r");
	  fflush(stdout);
	}
    }
  printf("trial %d ",n+1); 
  for (j=0;j<Npop;j++) printf("%g ",get_mean(A[j],Nbin));
  printf("\n");


  for (j=0;j<Npop;j++)
    for (i=0;i<Nbin/2;i++)  SA[j][i]/=Ntrials;


  gsl_rng_free (rng);
  free_dmatrix(A);
  free_imatrix(Abin);
  free_target_tensor(targets,Npop);
  free_neurons(neurons,Npop,p);
}



void get_psd_with_fullparameterlist(double **SA,int Nbin,int Ntrials, int Npop,double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[], double dt, double dtbin,int mode)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  struct Population p[Npop];

  double **signal=dmatrix(Npop,1);
  int i;
  for (i=0;i<Npop;i++) 
    signal[i]=NULL;

  DT=dt;
  DTBIN=dtbin;

  init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta, tau_theta, sigma, mode, DT);

  //  print_pop_parameters(p, Npop);

  free_dmatrix(signal);

  get_psd(SA,Nbin,Ntrials,p,Npop,dtbin,mode);
  free_population(Npop,p);
}








void get_psd_srm_with_2D_arrays(int Nf, double SA[][Nf], int Ntrials, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double J_ref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  //convert 2D array to double**
  double **SA_tmp, *J_theta_ptr[Npop], *tau_theta_ptr[Npop];
  int i,j;
  int Nbin=2*Nf;
  SA_tmp=dmatrix(Npop,Nf);

  int indx=0;
  for (i=0;i<Npop;i++)
    {
      /* printf("N=%d\n",N[i]); */
      /* printf("Ntheta=%d\n",N_theta[i]); */
      J_theta_ptr[i]=&(J_theta[indx]);
      tau_theta_ptr[i]=&(tau_theta[indx]);
      indx+=N_theta[i];
    }


  get_psd_with_fullparameterlist(SA_tmp, Nbin, Ntrials, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, N_theta, J_ref, J_theta_ptr, tau_theta_ptr, sigma, dt, dtbin, mode);


  for (j=0;j<Npop;j++) 
    for (i=0;i<Nf;i++)
      SA[j][i]=SA_tmp[j][i];

  free_dmatrix(SA_tmp);
}




void get_isih(double **isih, int Nspikes,struct Population p[],int Npop, double dtbin,int mode)
{
  gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2);
  unsigned long int seed;
  seed = random_seed();
  gsl_rng_set(rng,seed);
  //  gsl_rng_set (rng,(long)time(NULL));

  struct Neuron **neurons;
  neurons=initialize_neurons(Npop,p);

  int N=p[0].Nisi*dtbin;
  int **Abin;
  double **A;
  Abin=imatrix(Npop,N);
  A=dmatrix(Npop,N);
  int i,j;
  for (j=0;j<Npop;j++)
    {
      p[j].spikecount=0;
      p[j].isih=ivector(p[j].Nisi);
      for (i=0;i<p[j].Nisi;i++) p[j].isih[i]=0;
    }
  
  unsigned short ***targets;
  targets=allocate_target_tensor(Npop);
  
  construct_random_connectivity(targets,p,neurons,Npop,rng);

  //find max population size
  int Nmax=p[0].N;
  for (i=0;i<Npop;i++)
    {
      if (p[i].N>Nmax) Nmax=p[i].N;
    }

  //warmup
  int min=0,Nbin=0;
  while (min<10*Nmax)
    {
      simulate(Abin,N,neurons,p,targets,Npop,rng,mode,0);
      min=p[0].spikecount;
      for (j=1;j<Npop;j++)
	if (p[j].spikecount<min) min=p[j].spikecount;
      Nbin+=N;
    } 
    
  if (Nspikes>10*Nmax)   Nbin=(double)(Nbin)/(10*Nmax)*Nspikes;
  printf("Simulation time: %g sec\n",Nbin*dtbin);

  for (j=0;j<Npop;j++)
    {
      p[j].spikecount=0;
      for (i=0;i<p[j].Nisi;i++) p[j].isih[i]=0;
    }
  

  //start actual simulation
  free_dmatrix(A);
  free_imatrix(Abin);
  Abin=imatrix(Npop,Nbin);
  A=dmatrix(Npop,Nbin);

  simulate(Abin,Nbin,neurons,p,targets,Npop,rng,mode,0);
  for (j=0;j<Npop;j++)
    {
      p[j].spikecount=0;
      for (i=0;i<p[j].Nisi;i++) p[j].isih[i]=0;
    }
  simulate(Abin,Nbin,neurons,p,targets,Npop,rng,mode,0);

  //extract isi data
  for (j=0;j<Npop;j++)
    {
      int sum=0;
      double m1=0,m2=0;
      for (i=0;i<p[j].Nisi;i++)
	{
	  double t=(i+0.5)*dtbin;
	  sum+=p[j].isih[i];
	  isih[j][i]=(double)(p[j].isih[i])/(p[j].spikecount * dtbin);
	  m1+=t*isih[j][i];
	  m2+=t*t*isih[j][i];
	}
      m1*=dtbin;
      m2*=dtbin;
      double variance=m2-m1*m1;
      double rate=1./m1;
      double cv=sqrt(variance)/m1;
      double rest=(double)(p[j].spikecount-sum)/p[j].spikecount;
      printf("pop %d: rate=%g cvISI=%g Nspikes=%d rest=%g\n",j+1,rate,cv,p[j].spikecount,rest*100);
    }


  gsl_rng_free (rng);
  free_dmatrix(A);
  free_imatrix(Abin);
  free_target_tensor(targets,Npop);
  free_neurons(neurons,Npop,p);
  for (i=0;i<Npop;i++) free_ivector(p[i].isih);
}



void get_isih_with_fullparameterlist(double **isih,int Nisi,int Nspikes, int Npop,double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[], double dt, double dtbin,int mode)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  struct Population p[Npop];

  double **signal=dmatrix(Npop,1);
  int i;
  for (i=0;i<Npop;i++) 
    signal[i]=NULL;

  DT=dt;
  DTBIN=dtbin;

  init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta, tau_theta, sigma, mode, DT);

  for (i=0;i<Npop;i++) p[i].Nisi=Nisi; //set flag to measure isih

  //  print_pop_parameters(p, Npop);

  free_dmatrix(signal);

  get_isih(isih,Nspikes,p,Npop,dtbin,mode);
  free_population(Npop,p);
}


void get_isih_with_2D_arrays(int Nisi, double isih[][Nisi], int Nspikes, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double J_ref[], double J_theta[], double tau_theta[], double sigma[], double dt, double dtbin, int mode)
// ISI T[0..Nisi-1]
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  //convert 2D array to double**
  double **isih_tmp, *J_theta_ptr[Npop], *tau_theta_ptr[Npop];
  int i,j;
  isih_tmp=dmatrix(Npop,Nisi);

  int indx=0;
  for (i=0;i<Npop;i++)
    {
      /* printf("N=%d\n",N[i]); */
      /* printf("Ntheta=%d\n",N_theta[i]); */
      J_theta_ptr[i]=&(J_theta[indx]);
      tau_theta_ptr[i]=&(tau_theta[indx]);
      indx+=N_theta[i];
    }


  get_isih_with_fullparameterlist(isih_tmp, Nisi, Nspikes, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, N_theta, J_ref, J_theta_ptr, tau_theta_ptr, sigma, dt, dtbin, mode);


  for (j=0;j<Npop;j++) 
    for (i=0;i<Nisi;i++)
      isih[j][i]=isih_tmp[j][i];

  free_dmatrix(isih_tmp);
}

/* void get_isistat_with_fullparameterlist(double rate[], double cv[], double **isih, int Nbin,int Nspikes, int Npop,double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double dt, double dtbin,int mode) */
/* { */
  /* if (mode>=GLIF) mode=GLIF; */
  /* else mode=GLM; */

/*   struct Population p[Npop]; */

/*   double **signal=dmatrix(Npop,1); */
/*   int i,j; */
/*   for (i=0;i<Npop;i++)  */
/*     signal[i]=NULL; */

/*   init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, mu, c, deltaV, delay, vth, N, J, p_conn, signal, N_theta, J_ref, J_theta, tau_theta); */

/*   free_dmatrix(signal); */

/*   get_isistat(rate,cv,isih,Nbin,Nspikes,p,Npop,dt,dtbin,mode); */
/* } */







void setup_simulation_method(void (* update_neuron_state[])(struct Neuron *,struct Population *,int, int, gsl_rng *),  void (* receive_spike[])(struct Neuron *,struct Population *,int), int Npop, char psc_type[Npop][30])
{
  int j;

  for (j=0;j<Npop;j++)
    {
      if (strcmp(psc_type[j],"delta")==0)
	{
	  //	  printf("Use methods for %s psc\n",psc_type[j]);
	  update_neuron_state[j]=update_neuron_state_pscdelta;
	  receive_spike[j]=receive_spike_pscdelta;
	}
      else if (strcmp(psc_type[j],"exponential")==0)
	{
	  //	  printf("Use methods for %s psc\n",psc_type[j]);
	  update_neuron_state[j]=update_neuron_state_pscexponential;
	  receive_spike[j]=receive_spike_pscexponential;
	}
      else
	{
	  //	  printf("Use %s methods for psc\n",psc_type[j]);
	  update_neuron_state[j]=update_neuron_state_pscgeneric;
	  receive_spike[j]=receive_spike_pscgeneric;
	}
    }
}

















////////////////////////////////////////////////////////////////////////////////
//  trajectory with recording voltage traces
////////////////////////////////////////////////////////////////////////////////


void record_voltage(int k, double **voltmeter, int Nrecord[], int Nrec_tot, struct Neuron **neurons, struct Population p[], int Npop)
{
  int i,indx=0,j,n;
  for (i=0;i<Npop;i++)
    {
      for (j=0;j<Nrecord[i];j++)
	//record first Nrecord[i] neurons
	{
	  
	  if (neurons[i][j].tlast >= DTBIN + DT)
	    voltmeter[indx][k]=neurons[i][j].h + neurons[i][j].eta;
	  else //if there was a spike in last bin set voltage to large value
	    voltmeter[indx][k]=VSPIKE;

	  double theta_tot=0;
	  for (n=0;n<p[i].N_theta;n++) theta_tot+=neurons[i][j].theta[n];
	  voltmeter[indx+Nrec_tot][k]=theta_tot;
	  indx++;
	} 
    }
}




void simulate_with_voltage(int **Abin,int Tbin, double **voltmeter, int Nrecord[], struct Neuron **neurons,struct Population p[],unsigned short ***targets,int Npop,gsl_rng *rng,int mode, int dispprog)
//adds spikes to Abin[0..Tbin-1]
{
  int n,k,A[Npop],j;
  int nbin=DTBIN/DT+0.5;
  for (j=0;j<Npop;j++) A[j]=0;
  char psc_type[Npop][30];
  int Nrec_tot=0;
  for (j=0;j<Npop;j++)
    Nrec_tot+=Nrecord[j];


  init_internal_population_variables(p, Npop, psc_type);

  void (* update_neuron_state[Npop])(struct Neuron *,struct Population *,int, int, gsl_rng *);
  void (* receive_spike[Npop])(struct Neuron *, struct Population *, int);

  setup_simulation_method(update_neuron_state, receive_spike, Npop, psc_type);

  //step size for which to display progress
  int dispcount=(int)(Tbin/100);
  if (dispcount==0) dispcount=Tbin + 1;

  for (k=0;k<Tbin;k++)
    {
      for (n=0;n<nbin;n++)
	{
	  transfer_spikes(neurons,p,targets,Npop,receive_spike);
	  for (j=0;j<Npop;j++) update_neuron_state[j](neurons[j],p+j,k, mode, rng);
	  for (j=0;j<Npop;j++) get_spikes_glm(neurons[j],A+j,p+j,mode, rng);
	}

      for (j=0;j<Npop;j++) 
	{
	  Abin[j][k]=A[j];
	  A[j]=0;
	}
      record_voltage(k,voltmeter,Nrecord,Nrec_tot,neurons,p,Npop);

      if (dispprog)
	if ((k+1)%dispcount==0)
	  {
	    printf("%d%%  \r",(int)((k+1) * 100. / Tbin));
	    fflush(stdout);
	  }
    }

  free_internal_population_variables(p, Npop);
}

void get_trajectory_voltage_with_fullparameterlist(double **A,int Nbin,double **voltmeter, int Nrecord[], double V_spike, int Npop,double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[], double dt, double dtbin,int mode, int seed, int seed_quenched)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  DT=dt;
  DTBIN=dtbin;
  VSPIKE=V_spike;

  struct Population p[Npop];
  init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta, tau_theta, sigma, mode, DT);
  //  print_pop_parameters(p, Npop);

  gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set (rng,(long)seed);

  gsl_rng *rng_quenched=gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set (rng_quenched,(long)seed_quenched);

  //  printf("seed=%d X=%g\n",seed, gsl_rng_uniform(rng));
  
  struct Neuron **neurons;
  neurons=initialize_neurons(Npop,p);

  int **Abin;
  Abin=imatrix(Npop,Nbin);
  int i,j;

  unsigned short ***targets;
  targets=allocate_target_tensor(Npop);

  construct_random_connectivity(targets,p,neurons,Npop,rng_quenched);

  simulate_with_voltage(Abin,Nbin,voltmeter,Nrecord,neurons,p,targets,Npop,rng,mode,1);


  for (j=0;j<Npop;j++)
    for (i=0;i<Nbin;i++)
      A[j][i]=(double)(Abin[j][i])/dtbin/p[j].N;

  for (j=0;j<Npop;j++) printf("%g ",get_mean(A[j],Nbin));
  printf("\n");

  gsl_rng_free (rng);
  gsl_rng_free (rng_quenched);
  free_imatrix(Abin);
  free_target_tensor(targets,Npop);
  free_neurons(neurons,Npop,p);
  free_population(Npop,p);
}

void get_trajectory_voltage_srm_with_2D_arrays(int Nbin, double AA[][Nbin], double voltmeter[][Nbin], int Nrecord[], double V_spike, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double s[][Nbin], int N_theta[], double J_ref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode, int seed, int seed_quenched)
// input: voltmeter is a 2d array of shape (sum_{i=0}^{Npop-1} Nrecord[i],Nbin)
{
  if (mode>=GLIF) mode=GLIF;
  else mode=GLM;

  //convert 2D array to double**
  double *AAA[Npop], *signal[Npop], *J_theta_ptr[Npop], *tau_theta_ptr[Npop];
  int i;
  int Nrec_tot=0;
  for (i=0;i<Npop;i++)
    Nrec_tot+=Nrecord[i];

  double *voltage[2*Nrec_tot];

  for (i=0;i<Npop;i++)
    AAA[i]=AA[i];
  for (i=0;i<2*Nrec_tot;i++)
    voltage[i]=voltmeter[i];
  for (i=0;i<Npop;i++)
    signal[i]=s[i];


  int indx=0;
  for (i=0;i<Npop;i++)
    {
      /* printf("N=%d\n",N[i]); */
      /* printf("Ntheta=%d\n",N_theta[i]); */
      if (N_theta[i]>0)
  	{
  	  J_theta_ptr[i]=&(J_theta[indx]);
  	  tau_theta_ptr[i]=&(tau_theta[indx]);
  	  indx+=N_theta[i];
  	}
      else
  	{
  	  J_theta_ptr[i]=NULL;
  	  tau_theta_ptr[i]=NULL;
  	}
    }


  get_trajectory_voltage_with_fullparameterlist(AAA, Nbin, voltage, Nrecord, V_spike, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta_ptr, tau_theta_ptr, sigma, dt, dtbin, mode, seed, seed_quenched);
}







/* //////////////////////////////////////////////////////////////////////////////// */
/* //  ISI statistics */
/* //////////////////////////////////////////////////////////////////////////////// */


/* void get_spikes_srm_sfa_isi(struct Neuron *neurons,double *m1, double *m2, int *isih, int *spikecount,struct Population *p,gsl_rng *rng,int Nbin) */
/* { */
/*   int i,k,n=p->N,ndelay=p->delay/DT; */
/*   double rho; */
/*   for (i=0;i<n;i++) */
/*     { */
/*       if (gsl_rng_uniform(rng)<=pspike_srm_sfa(neurons+i,p)) */
/* 	{ */
/* 	  neurons[i].axon[neurons[i].offset]=1; // send spike into axon */

/* 	  *m1+=neurons[i].tlast; */
/* 	  *m2+=neurons[i].tlast*neurons[i].tlast; */
/* 	  /\* if (neurons[i].tlast>0.5) *\/ */
/* 	  /\*   printf("isi=%g\n",neurons[i].tlast); *\/ */
/* 	  int indx=neurons[i].tlast/DTBIN; */
/* 	  if (indx<Nbin) isih[indx]++; */
/* 	  (*spikecount)++; */

/* 	  neurons[i].tlast=0; */

/* 	  neurons[i].eta_ref = p->J_ref; */

/* 	  for (k=0; k < p->N_theta; k++)  */
/* 	    neurons[i].theta[k]+=p->J_theta[k]; */
/* 	} */
/*       neurons[i].offset=(neurons[i].offset+1)%ndelay; */
/*       neurons[i].tlast+=DT; */
/*     } */
/* } */

/* void get_spikes_lif_nosfa_isi(struct Neuron *neurons,double *m1, double *m2, int *isih, int *spikecount,struct Population *p,gsl_rng *rng,int Nbin) */
/* { */
/*   int i,n=p->N,ndelay=p->delay/DT; */
/*   double rho; */
/*   for (i=0;i<n;i++) */
/*     { */
/*       if (gsl_rng_uniform(rng)<=pspike(neurons+i,p)) */
/* 	{ */
/* 	  neurons[i].axon[neurons[i].offset]=1; // send spike into axon */

/* 	  *m1+=neurons[i].tlast; */
/* 	  *m2+=neurons[i].tlast*neurons[i].tlast; */
/* 	  int indx=neurons[i].tlast/DTBIN; */
/* 	  if (indx<Nbin) isih[indx]++; */
/* 	  (*spikecount)++; */

/* 	  neurons[i].v=0; */
/* 	  neurons[i].tlast=0; */
/* 	} */
/*       neurons[i].offset=(neurons[i].offset+1)%ndelay; */
/*       neurons[i].tlast+=DT; */
/*     } */
/* } */

/* void get_spikes_lif_sfa_isi(struct Neuron *neurons,double *m1, double *m2, int *isih, int *spikecount,struct Population *p,gsl_rng *rng,int Nbin) */
/* { */
/*   int i,k,n=p->N,ndelay=p->delay/DT; */
/*   double rho; */
/*   for (i=0;i<n;i++) */
/*     { */
/*       if (gsl_rng_uniform(rng)<=pspike_lif_sfa(neurons+i,p)) */
/* 	{ */
/* 	  neurons[i].axon[neurons[i].offset]=1; // send spike into axon */
/* 	  *m1+=neurons[i].tlast; */
/* 	  *m2+=neurons[i].tlast*neurons[i].tlast; */
/* 	  int indx=neurons[i].tlast/DTBIN; */
/* 	  if (indx<Nbin) isih[indx]++; */
/* 	  (*spikecount)++; */
/* 	  neurons[i].v=0; */
/* 	  neurons[i].tlast=0; */
/* 	  for (k=0; k < p->N_theta; k++)  */
/* 	    neurons[i].theta[k]+=p->J_theta[k]; */
/* 	} */
/*       neurons[i].offset=(neurons[i].offset+1)%ndelay; */
/*       neurons[i].tlast+=DT; */
/*     } */
/* } */


/* void setup_simulation_method_isi(void (* update_neuron_state[])(struct Neuron *,struct Population *,int),  void (* get_spikes[])(struct Neuron *,double *,double *,int *,int *,struct Population *,gsl_rng *,int), struct Population p[], int Npop, int Ns,int mode) */
/* { */
/*   int j; */
/*   for (j=0;j<Npop;j++) */
/*     { */
/*       if (mode=="srm") */
/* 	{ */
/* 	  switch (Ns) */
/* 	    { */
/* 	    case 0: */
/* 	      update_neuron_state[j]=update_neuron_state_srm_psc_delta_sfa; */
/* 	      break; */
/* 	    case 1: */
/* 	      update_neuron_state[j]=update_neuron_state_srm_psc_exp_sfa; */
/* 	      break; */
/* 	    case 2: */
/* 	      update_neuron_state[j]=update_neuron_state_srm_psc_twoexp_sfa; */
/* 	    } */
/* 	  get_spikes[j]=get_spikes_srm_sfa_isi; */
/* 	} */
/*       else // if (mode=="lif") */
/* 	{ */
/* 	  if (p[j].N_theta + p[j].N_adap == 0) */
/* 	    //no adaptation */
/* 	    {       */
/* 	      switch (Ns) */
/* 		{ */
/* 		case 0: */
/* 		  update_neuron_state[j]=update_neuron_state_lif_psc_delta; */
/* 		  break; */
/* 		case 1: */
/* 		  update_neuron_state[j]=update_neuron_state_lif_psc_exp; */
/* 		  break; */
/* 		case 2: */
/* 		  update_neuron_state[j]=update_neuron_state_lif_psc_twoexp; */
/* 		} */
	      
/* 	      get_spikes[j]=get_spikes_lif_nosfa_isi; */
/* 	    } */
/* 	  else  */
/* 	    { */
/* 	      switch (Ns) */
/* 		{ */
/* 		case 0: */
/* 		  update_neuron_state[j]=update_neuron_state_lif_psc_delta_sfa; */
/* 		  break; */
/* 		case 1: */
/* 		  update_neuron_state[j]=update_neuron_state_lif_psc_exp_sfa; */
/* 		  break; */
/* 		case 2: */
/* 		  update_neuron_state[j]=update_neuron_state_lif_psc_twoexp_sfa; */
/* 		} */

/* 	      get_spikes[j]=get_spikes_lif_sfa_isi; */
/* 	    } */
/* 	} */
/*     } */

/* } */


/* void simulate_isistat(double rate[], double cv[], double **isih, int Nbin, int Nspikes,struct Neuron **neurons,struct Population p[],unsigned short ***targets,int Npop,gsl_rng *rng,int mode) */
/* //adds spikes to Abin[0..Tbin-1] */
/* { */
/*   int n,k,j; */
/*   double t=0; */
/*   int Ns=0; //number of exponentials */
/*   int h[Npop][Nbin],spikecount[Npop]; */
/*   double m1[Npop], m2[Npop];  //1st and 2nd moment of ISI */
/*   for (k=0;k<Npop;k++) */
/*     { */
/*       for (j=0;j<Nbin;j++) h[k][j]=0; */
/*       m1[k]=0; */
/*       m2[k]=0; */
/*       spikecount[k]=0; */
/*     } */

/*   init_internal_population_variables(p, Npop, &Ns); */

/*   void (* update_neuron_state[Npop])(struct Neuron *,struct Population *,int); */
/*   void (* get_spikes[Npop])(struct Neuron *,double *, double *, int *, int *, struct Population *,gsl_rng *,int); */

/*   setup_simulation_method_isi(update_neuron_state, get_spikes, p, Npop, Ns, mode); */

/*   int min=0; */
/*   while (min<Nspikes) */
/*     { */
/*       transfer_spikes(neurons,p,targets,Npop); */
/*       for (j=0;j<Npop;j++) update_neuron_state[j](neurons[j],p+j,k); */
/*       for (j=0;j<Npop;j++) get_spikes[j](neurons[j],m1+j,m2+j,h[j],spikecount+j,p+j,rng,Nbin); */
      
/*       min=spikecount[0]; */
/*       for (k=1;k<Npop;k++) */
/* 	if (spikecount[k]<min) min=spikecount[k]; */

/*       t+=DT; */
/*     } */


/*     printf("t=%g\n",t); */

/*   for (k=0;k<Npop;k++)  */
/*     { */
/*       m1[k]/=spikecount[k]; */
/*       m2[k]/=spikecount[k]; */
/*       rate[k]=1./m1[k]; */
/*       double var=m2[k]-m1[k]*m1[k]; */
/*       cv[k]=sqrt(var)/m1[k]; */
/*       for (j=0;j<Nbin;j++) */
/* 	isih[k][j]=(double)(h[k][j])/(DTBIN*spikecount[k]); */
/*       printf("pop %d: rate=%g CV=%g\n",k+1,rate[k],cv[k]); */
/*     } */


/*   free_internal_population_variables(p, Npop); */
 
/* } */



/* void get_isistat(double rate[], double cv[], double **isih, int Nbin,int Nspikes,struct Population p[],int Npop,double dt,double dtbin,int mode) */
/* { */
/*   gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2); */
/*   gsl_rng_set (rng,(long)time(NULL)); */

/*   DT=dt; */
/*   DTBIN=dtbin; */

/*   struct Neuron **neurons; */
/*   neurons=initialize_neurons(Npop,p); */

/*   unsigned short ***targets; */
/*   targets=allocate_target_tensor(Npop); */

/*   construct_random_connectivity(targets,p,neurons,Npop,rng); */

/*   int i,Nmax=p[0].N; */
/*   for (i=0;i<Npop;i++) */
/*     { */
/*       if (p[i].N>Nmax) Nmax=p[i].N; */
/*     } */
/*   //warmup */
/*   simulate_isistat(rate,cv,isih,Nbin,10*Nmax,neurons,p,targets,Npop,rng,mode); */

/*   //simulation of stationary sequence */
/*   simulate_isistat(rate,cv,isih,Nbin,Nspikes,neurons,p,targets,Npop,rng,mode); */

/*   gsl_rng_free (rng); */
/*   free_target_tensor(targets,Npop); */
/*   free_neurons(neurons,Npop,p); */
/* } */





void print_pop_parameters(struct Population p[], int Npop)
{

  int k,l;
  printf("\n");
  for (k=0;k<Npop;k++)
    {
      printf("POPULATION %d\n",k+1);
      printf("tref=%g\n",p[k].tref);
      printf("taum=%g\n",p[k].taum);
      printf("mu=%g\n",p[k].mu);
      printf("c=%g\n",p[k].c);
      printf("DeltaV=%g\n",p[k].deltaV);
      printf("delay=%g\n",p[k].delay);
      printf("vth=%g\n",p[k].vth);
      printf("vreset=%g\n",p[k].h_reset);
      printf("Jref=%g\n",p[k].J_ref);
      printf("vreset=%g\n",p[k].h_reset);
      printf("N=%d\n",p[k].N);
      printf("Ntheta=%d\n",p[k].N_theta);
      for (l=0;l<p[k].N_theta;l++)
	{
	  printf("Jtheta=%g mVs\n",p[k].theta_jump[l] * p[k].tau_theta[l]);
	  printf("tautheta=%g\n",p[k].tau_theta[l]);
	}  
      for (l=0;l<Npop;l++)
	{
	  printf("pop%d to pop%d:\n",l+1,k+1);
	  printf("   a1=%g\n",p[k].a1[l]);
	  printf("   a2=%g\n",p[k].a2[l]);
	  printf("   taus1=%g taur1=%g\n",p[k].taus1[l],p[k].taur1[l]);
	  printf("   taus2=%g taur2=%g\n",p[k].taus2[l],p[k].taur2[l]);
	}
      printf("\n");
    }

}

#include <sys/time.h>
unsigned long int random_seed()
{

 unsigned int seed;
 struct timeval tv;
 FILE *devrandom;

 if ((devrandom = fopen("/dev/urandom","r")) == NULL) {
   gettimeofday(&tv,0);
   seed = tv.tv_sec + tv.tv_usec;
 } else {
   fread(&seed,sizeof(seed),1,devrandom);
   fclose(devrandom);
 }

 return(seed);

}
