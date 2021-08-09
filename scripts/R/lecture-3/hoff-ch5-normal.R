
# Example from Hoff Chapter 5 on the normal distribution

library(reshape)

y = c(1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08)
n = length(y)
ybar = mean(y)
s2 = var(y)

mu0 = 1.9
kappa0 = 1
s20 = 0.01
nu0 = 1

kappan = kappa0 + n
nun = nu0 + n
mun = (kappa0 * mu0 + n * ybar) / kappan
s2n = (1 / nun) * (nu0 * s20 + (n - 1) * s2 + (kappa0 * n / kappan) * (ybar - mu0)^2)

Theta = seq(1.6, 2.0, by = 0.005)
Sigma2 = seq(0, 0.04, by = 0.0001)

library(invgamma)
post.func = function(theta, sigma2) {
  dnorm(theta, mun, sqrt(sigma2 / kappan)) * dinvgamma(sigma2, nun / 2, s2n * nun / 2)
}

d = outer(Theta, Sigma2, post.func)
rownames(d) = Theta
colnames(d) = Sigma2

df = melt(d)
colnames(df) = c('theta', 'sigma2', 'density')

ggplot(df, aes(x = theta, y = sigma2, z = density)) +
  geom_contour(aes(color = ..level..)) +
  guides(color = FALSE)

# Monte Carlo sampling

s2.mc = rinvgamma(10000, nun / 2, s2n * nun / 2)
theta.mc = rnorm(10000, mun, sqrt(s2.mc / kappan)) # Accepts a vector of parameters
mean(theta.mc)

quantile(theta.mc, c(0.025, 0.975))

ggplot(data.frame(sigma2 = s2.mc, theta = theta.mc)) +
  geom_point(aes(x = theta, y = sigma2), alpha = 0.3)

