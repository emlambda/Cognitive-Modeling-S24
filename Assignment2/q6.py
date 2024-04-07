import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import stan
import nest_asyncio
nest_asyncio.apply()
#yn ∼ N (α + β xn, σ2) for n = 1, . . . , N

stan_code = """
data {
  int<lower=1> T;         // Tumber of observations
  vector[T] x;            // Covariate values
  vector[T] y;            // Outcome values
}

parameters {
  real alpha;             // Intercept
  real beta;              // Slope
  real<lower=0> sigma;    // Toise standard deviation
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ inv_gamma(1, 1);
  
  // Likelihood
  for (t in 2:T){
    y[t] ~ normal(alpha + beta * x[t], sigma);
  }
}
generated quantities {
  array[T] real y_pred;
  y_pred[1] = 0;
  for (t in 2:T) {
    y_pred[t] = normal_rng(alpha + beta * x[t], sigma);
  }
}
"""

T = 100
alpha_true = 2.3
beta_true = 4.0
sigma_true = 2.0
x = np.random.normal(size=T)
y = alpha_true + beta_true * x + sigma_true * np.random.normal(size=T)

# Prepare data for Stan
stan_data = {'T': T, 'x': x, 'y': y}

posterior_samples = stan.build(stan_code,data=stan_data)

fit = posterior_samples.sample(num_samples=20000, num_chains=4, num_warmup=500)

# Summarize posterior
print(fit)

ax_=az.plot_forest(fit, var_names=['alpha','beta','sigma'])
ax_[0].sharey = True
ax_[0].scatter([alpha_true, beta_true, sigma_true],ax_[0].get_yticks(),marker='o',c='r',label='actual values')
ax_[0].legend()

summary = az.summary(fit)
print(summary)

plt.scatter([1,2,3],summary[:3]['mean'], label='posterior means')
plt.scatter([1,2,3],[alpha_true, beta_true, sigma_true],label='actual values')
plt.xticks([1,2,3],['alpha','beta','sigma'])
plt.legend()

uncertainty = summary[:3]['sd']/[alpha_true, beta_true, sigma_true]

df = fit.to_frame()

fig,ax = plt.subplots(sharex=True,sharey=True) 

ax.plot(df.iloc[:,df.shape[1]-T:].values[:100,:].T, color='maroon', alpha=0.1)
ax.plot(y, 'o-', color='black',label = 'actual')
ax.set_xlabel('T')
ax.set_ylabel('y')
ax.legend()
