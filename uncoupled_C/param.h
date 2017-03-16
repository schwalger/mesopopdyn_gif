#define NPOP 1
static int N[NPOP] = {1000};
static double mu[NPOP] = {18.0};
static double DeltaV[NPOP] = {2.0};
static double c[NPOP] = {10.0};
static double tref[NPOP] = {0.004};
static double delay[NPOP] = {0.001};
static double vth[NPOP] = {15.0};
static double vreset[NPOP] = {0.0};
static double taum[NPOP] = {0.02};
static double sigma[NPOP] = {0.0};
static double taus1[NPOP][NPOP] = {{0.0}};
static double taus2[NPOP][NPOP] = {{0.0}};
static double taur1[NPOP][NPOP] = {{0.0}};
static double taur2[NPOP][NPOP] = {{0.0}};
static double a1[NPOP][NPOP] = {{1.0}};
static double a2[NPOP][NPOP] = {{0.0}};
static double J[NPOP][NPOP] = {{0.0}};
static double p_conn[NPOP][NPOP] = {{1.0}};
static int N_theta[NPOP]={0};
static double J_ref[NPOP]={0.0};


void fill_fbkernel(double *J_theta[], double *tau_theta[]){
static double J_theta0[0]={};
J_theta[0] = J_theta0;
static double tau_theta0[0]={};
tau_theta[0] = tau_theta0;
}


char paramstr[]="_mode10_Npop1_mu18_du2_vth15_vr0_c10_J0_p1_taus1_0_taum0.02_N1_1000_delay0.001_tref0.004_Na1_Ja0_taua1_sigma0";
