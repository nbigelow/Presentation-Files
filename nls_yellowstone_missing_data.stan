functions{
 real[] dz_dt(real t, real[] z, real[] theta,real[] x_r, int[] x_i) {
  real x = z[1];
  real y = z[2];

  real alpha = theta[1];
  real beta = theta[2];
  real gamma = theta[3];
  real delta = theta[4];

  real dx_dt = (alpha - beta * y) * x;
  real dy_dt = (-gamma + delta * x) * y;
  return {dx_dt, dy_dt};
 }
}

data {
  int N;                    // Length of entire data set
  int M;                    // Number of predictions
  real<lower=0> y_init[2]; // Initial values
  real time[N];           // total time to be integrated
  real ts_new[M];             // Times of predicted values
  // Observed Data
  int J;                  // num observations 
  int tt[J];             // time for obs j
  real<lower = 0> elk[J];  // 
  real<lower = 0> wolves[J];  // 
  // Missing Data
  int J_miss;             // num missing observations
  int tm[J_miss];        // missing time for obs j
}

parameters {
  real<lower = 0,upper = 1> theta[4];   // theta = { alpha, beta, gamma, delta }
  real<lower = 0> z_init[2];  // initial population
  real<lower = 0> elk_missing[J_miss];  //  // missing obs
  real<lower = 0> wolves_missing[J_miss];  //  // missing obs
  real<lower = 0> sigma[2];   // error scale
}

// Solves the ODE's & returns solutions
transformed parameters {
  real zz[N, 2] = integrate_ode_bdf(dz_dt, z_init, 0, time, theta,rep_array(0.0, 0), rep_array(0, 0),1e-6, 1e-5, 1e10);
}

model {
  // Priors
  theta[{1, 3}] ~ beta(1,1);
  theta[{2, 4}] ~ beta(1, 5);
  sigma ~ gamma(1, 5);
  z_init[1] ~ lognormal(log(y_init[1]), sigma);
  z_init[2] ~ lognormal(log(y_init[2]), sigma);

  // Liklihood obs
 for (j in 1:J){
    elk[j] ~ lognormal(log(zz[tt[j], 1]), sigma[1]);
    wolves[j] ~ lognormal(log(zz[tt[j], 2]), sigma[2]);
 }
 // Likelihood missing
 for(k in 1:J_miss){
   elk_missing[k] ~ lognormal(log(zz[tm[k],1]),sigma[1]);
   wolves_missing[k] ~ lognormal(log(zz[tm[k],2]),sigma[2]);
 }
}

generated quantities{
  // Sample from missing values
  real<lower=0> y_rep_elk_miss[2];
  real<lower=0> y_rep_wolves_miss[2];
  real<lower=0> y_rep[J,2];
  real z_pred[M, 2] = integrate_ode_bdf(dz_dt, zz[N,], time[N], ts_new, theta,rep_array(0.0, 0), rep_array(0, 0),1e-6, 1e-5, 1e10);
  real y_pred[M,2]; // Array of predictions
  for(l in 1:J_miss){
    y_rep_elk_miss[l] = lognormal_rng(log(elk_missing[l]),sigma[1]);
    y_rep_wolves_miss[l] = lognormal_rng(log(wolves_missing[l]),sigma[2]);
  }
  // Sample from data fits
  for(m in 1:J){
    y_rep[m,1] = lognormal_rng(log(elk[m]),sigma[1]);
    y_rep[m,2] = lognormal_rng(log(wolves[m]),sigma[2]);
  }
  // Predicted quantities
  // Solution given the final time step z[N,]
  for(k in 1:2){
     y_pred[ , k] = lognormal_rng(log(z_pred[, k]), sigma[k]); // Generate predicted value
  }
}
