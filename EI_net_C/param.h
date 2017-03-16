#define NPOP 2
static int N[NPOP] = {800, 200};
static double mu[NPOP] = {24.0, 24.0};
static double DeltaV[NPOP] = {2.5, 2.5};
static double c[NPOP] = {10.0, 10.0};
static double tref[NPOP] = {0.004, 0.004};
static double delay[NPOP] = {0.001, 0.001};
static double vth[NPOP] = {15.0, 15.0};
static double vreset[NPOP] = {0.0, 0.0};
static double taum[NPOP] = {0.02, 0.02};
static double sigma[NPOP] = {0.0, 0.0};
static double taus1[NPOP][NPOP] = {{0.003, 0.006}, {0.003, 0.006}};
static double taus2[NPOP][NPOP] = {{0.0, 0.0}, {0.0, 0.0}};
static double taur1[NPOP][NPOP] = {{0.0, 0.0}, {0.0, 0.0}};
static double taur2[NPOP][NPOP] = {{0.0, 0.0}, {0.0, 0.0}};
static double a1[NPOP][NPOP] = {{1.0, 1.0}, {1.0, 1.0}};
static double a2[NPOP][NPOP] = {{0.0, 0.0}, {0.0, 0.0}};
static double J[NPOP][NPOP] = {{0.3, -1.5}, {0.3, -1.5}};
static double p_conn[NPOP][NPOP] = {{0.2, 0.2}, {0.2, 0.2}};
static int N_theta[NPOP]={0, 0};
static double J_ref[NPOP]={0.0, 0.0};


void fill_fbkernel(double *J_theta[], double *tau_theta[]){
static double J_theta0[0]={};
J_theta[0] = J_theta0;
static double tau_theta0[0]={};
tau_theta[0] = tau_theta0;
static double J_theta1[0]={};
J_theta[1] = J_theta1;
static double tau_theta1[0]={};
tau_theta[1] = tau_theta1;
}


char paramstr[]="_mode10_Npop2_mu24_du2.5_vth15_vr0_c10_J1_0.3_J2_-1.5_p1_0.2_p2_0.2_taus1_0.003_taus2_0.006_taum0.02_N1_800_N2_200_delay0.001_tref0.004_Na1_Ja0_taua0_sigma0";
