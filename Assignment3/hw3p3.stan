// MPT_model.stan
data {
  int<lower=0> N; 
  int<lower=1> C; 
  int<lower=1, upper=C> y[N]; 
}

parameters {
  real<lower=0, upper=1> a;
  real<lower=0, upper=1> b; 
  real<lower=0, upper=1> c; 

}

transformed parameters {
  simplex[C] p; 
}

model {

  a ~ beta(1, 1);
  b ~ beta(1, 1);
  c ~ beta(1, 1);


  for (n in 1:N) {
    y[n] ~ categorical(p);
  }
}

generated quantities {
  int<lower=1, upper=C> y_new[N];
  for (n in 1:N) {
    y_new[n] = categorical_rng(p);
  }
}
