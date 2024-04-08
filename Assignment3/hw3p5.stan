data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    vector[N] y;
}
parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
}
model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 5);
    sigma ~ inv_gamma(2, 3);

    // Likelihood
    y ~ normal(X * beta + alpha, sigma);
}