//treats refractory groups and free group separately
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define PI 3.141592653
/* #define GLM 0 */
/* #define GLIF 10//default glif update */
/* #define GLIFMASTER 30//default glif update */




#include PARAM

//#include "../glm_netw_sim_0.7.c"
#include "../glm_netw_sim_0.8.c"

#define dt Dt
#define dtbin Dtbin









int main(void)
{
  int Nbin=T_sim/dtbin;

  //create external stimulus
  double **signal=dmatrix(NPOP,Nbin);
  int i,j;
  for (j=0;j<NPOP;j++)
    signal[j]=NULL;
  /*   for (i=0;i<Nbin;i++) signal[j][i]=0; */

  /* double t_inj=2.;  //injection time */
  /* for (i=0;i<Nbin;i++) */
  /*   if (i>=(int)(t_inj/dtbin)) signal[0][i]=0.; */


  double *J_theta[NPOP], *tau_theta[NPOP];
  fill_fbkernel(J_theta,tau_theta);

  /* //create external stimulus */
  /* double fs=1.,eps=10.; */
  /* double *signal=dvector(Nbin); */
  /* for (i=0;i<Nbin;i++) signal[i]=eps*sin(2*PI*fs*i*dtbin); */

  double **A, **aa;
  A=dmatrix(NPOP,Nbin);
  int seed=365;
  //  clock_t start=clock();

  //  void get_trajectory_with_fullparameterlist(double **A,int Nbin,int Npop,double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[], double dt, double dtbin,int mode,int seed,int seed_quenched)

  get_trajectory_with_fullparameterlist(A, Nbin, NPOP, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, DeltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, J_ref, J_theta, tau_theta, sigma, dt, dtbin, MODE,seed,seed);
  //  printf("Execution time: %g seconds\n",(double)(clock()-start)/CLOCKS_PER_SEC); 




  char buffer[300];
  char buf2[100];
  sprintf(buf2,"_dt%g_dtbin%g_mode%d.dat",dt,(int)(dtbin/dt)*dt,MODE);
  strcpy(buffer,"data/trajec_pop");
  strcat(buffer, paramstr);
  strcat(buffer, buf2);
  
  FILE *f;
  f=fopen(buffer,"w");

  for (i=0;i<Nbin;i++)
    {
      fprintf(f,"%g ",dtbin*i);
      for (j=0;j<NPOP;j++)
	fprintf(f,"%g ",A[j][i]);
      fprintf(f,"\n");
    }

  fclose(f);


  free_dmatrix(A);
  return 0;
}
