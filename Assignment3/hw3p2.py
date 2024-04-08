import stan
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

import nest_asyncio
nest_asyncio.apply()

y = [0,0,1,1,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1]
y_pred = [0,0,1,1,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,1]
N = len(y)

num_tp = sum(y[i]==y_pred[i] for i in range(N) if y[i]== 1)
num_tn = sum(y[i]==y_pred[i] for i in range(N) if y[i]== 0)
num_fp = sum(y[i]!=y_pred[i] for i in range(N) if y[i]== 0)
num_fn = sum(y[i]!=y_pred[i] for i in range(N) if y[i]== 1)
data = np.array([num_tp,num_tn,num_fp,num_fn])

one_high_thresh = """
data {
  int<lower=1> N;
  int<lower=1> K;
  array[K] int<lower=0, upper=N> freqs;
}

parameters {
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> g;
}

transformed parameters {
  simplex[K] theta;
  theta[1] = 0.5 * d + (0.5 * (1 - d) * g);     // true positive
  theta[2] = 0.5 * (1 - g);                     // true negative
  theta[3] = 0.5 * g;                           // false positive
  theta[4] = 0.5 * (1 - d) * (1 - g);           // false negative
}

model {
  target += beta_lpdf(d | 1, 1);
  target += beta_lpdf(g | 1, 1);
  target += multinomial_lpmf(freqs | theta);
}

generated quantities{
  array[K] int pred_freqs;
  pred_freqs = multinomial_rng(theta, N);
}
"""

stan_dict = {
    'N': N,
    'K': data.shape[0],
    'freqs': data
}

# Compile model
posterior = stan.build(one_high_thresh, data=stan_dict, random_seed=42)

# Sample (i.e., inverse inference)
fit = posterior.sample(num_chains=4, num_samples=2500, num_warmup=1000)

sns.histplot(fit.to_frame().iloc[:, -1], binwidth=0.5)

az.summary(fit)

ax = az.plot_trace(fit, var_names=[r'theta'], filter_vars='like', compact=False, legend=True, figsize=(16, 10))
plt.tight_layout()

print(f"Actual theta: {data/N}")

two_high_thresh = """
data {
  int<lower=1> N;
  int<lower=1> K;
  array[K] int<lower=0, upper=N> freqs;
}

parameters {
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> g;
}

transformed parameters {
  simplex[K] theta;
  theta[1] = 0.5 * d + (0.5 * (1 - d) * g);        // true positive
  theta[2] = 0.5 * d + (0.5 * (1 - d) * (1 - g));  // true negative
  theta[3] = 0.5 * (1 - d) * g;                    // false positive
  theta[4] = 0.5 * (1 - d) * (1 - g);              // false negative
}

model {
  target += beta_lpdf(d | 1, 1);
  target += beta_lpdf(g | 1, 1);
  target += multinomial_lpmf(freqs | theta);
}

generated quantities{
  array[K] int pred_freqs;
  pred_freqs = multinomial_rng(theta, N);
}
"""

stan_dict = {
    'N': N,
    'K': data.shape[0],
    'freqs': data
}

# Compile model
posterior = stan.build(two_high_thresh, data=stan_dict, random_seed=42)

# Sample (i.e., inverse inference)
fit = posterior.sample(num_chains=4, num_samples=2500, num_warmup=1000)

sns.histplot(fit.to_frame().iloc[:, -1], binwidth=0.5)

az.summary(fit)

ax = az.plot_trace(fit, var_names=[r'theta'], filter_vars='like', compact=False, legend=True, figsize=(16, 10))
plt.tight_layout()

print(f"Actual theta: {data/N}")