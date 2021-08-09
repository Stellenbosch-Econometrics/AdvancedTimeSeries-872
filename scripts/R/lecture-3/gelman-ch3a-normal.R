
## Figures for the slides from Gelman Ch 3. 

# Include the following packages
library(purrr)
library(dplyr)
library(tidyr)
library(rstan)
library(rstanarm)
options(mc.cores = parallel::detectCores())
library(ggplot2)
library(grid)
library(gridExtra)
library(bayesplot)
library(latex2exp)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))

# Simulate fake data

x <- 1:20
n <- length(x)
a <- 0.2
b <- 0.3
sigma <- 1

# set the random seed to get reproducible results
# change the seed to experiment with variation due to random noise
set.seed(2141) 

# Construct a model 
y <- a + b*x + sigma*rnorm(n)
fake <- data.frame(x, y)

# Fit a linear model using GLM to these data points (it appears that this has generated 4000 draws from the posterior for this fit)

# In essence we are drawing samples from the posterior distribution, with no prior specified and the likelihood function originating from the sampling model indicated above. 
fit_1 <- stan_glm(y ~ x, data = fake, seed=2141) 
print(fit_1, digits=2)

# Compare against the frequentist fit of the model with MLE
glm_fit <- glm(y ~ x, data = fake, family=gaussian)
summary(glm_fit)

# Extract the posterior draws from this fit
sims <- as.matrix(fit_1)
n_sims <- nrow(sims)

# Post predictive check
pp_check(fit_1)

# Compute predictive distribution given posterior mean and x=18
cond <- data.frame(y=seq(0,9,length.out=100))
cond$x <- dnorm(cond$y, mean(sims[,1]) + mean(sims[,2])*18, mean(sims[,3]))*6+18

# Plot the data
p1 <- ggplot(fake, aes(x = x, y = y)) +
  xlim(0,21) + ylim(0,9) +
  geom_point(color = "black", size = 2) +
  labs(title = "Data") 
p1

# Plot posterior mean
p2 <- p1 +
  geom_abline(
    intercept = mean(sims[, 1]),
    slope = mean(sims[, 2]),
    size = 1,
    color = "red"
  ) +
  labs(title = "Posterior mean") 
p2

# Plot all the posterior draws
p2s <- p2 +
  geom_abline(
    intercept = sims[seq(1,1001,by=10), 1],
    slope = sims[seq(1,1001,by=10), 2],
    size = 0.1,
    color = "blue",
    alpha = 0.2
  ) + ggtitle("Posterior draws")
p2s

# Simple fake univariate data
# set the random seed to get reproducible results
# change the seed to experiment with variation due to random noise
set.seed(2141) 
y <- sigma*rnorm(n)
fake <- data.frame(y)

# Fit a Gaussian
fit_1 <- stan_glm(y ~ 1, data = fake, seed=2141)

# Extract the posterior draws
sims <- as.matrix(fit_1)
n_sims <- nrow(sims)

# Plot the data
p1 <- ggplot(fake, aes(x = y, y = 0)) +
  ylim(0,0.6) + xlim(-4,4) +
  geom_point(color = "black", size = 2) +
  labs(title = "Data", y="") 
p1

# Plot a Gaussian with posterior mean parameters
p2 <- p1 +
  stat_function(fun = dnorm, n = 101,
                args = list(mean = mean(sims[,1]), sd = mean(sims[,2]))) +
  labs(title = "Gaussian fit with posterior mean", y="density") 
p2

muhat <- mean(sims[,1]) # Estimate of what mu is, mean of the mean estimates
sigmahat <- mean(sims[,2]) # Estimate of what sigma is, mean of the sigma estimates

p2mu <- p2 +
  geom_segment(aes(x=0, xend=0, y=0, yend=dnorm(0, 0, sigmahat)),
               linetype=2) +
  annotate(geom = "text", x=0.2, y=0.05, label=TeX("$\\hat{\\mu}$"), size=6)

p2musd <- p2mu +
  geom_segment(aes(x=0, xend=sigmahat,
                   y=dnorm(sigmahat, muhat, sigmahat),
                   yend=dnorm(sigmahat, muhat, sigmahat)),
               linetype=2) +
  annotate(geom = "text", x=0.5, y=0.22, label=TeX("$\\hat{\\sigma}$"),
           size=6)
p2musd

# Plot the posterior draws from fit_1
draws <- as.data.frame(fit_1)[seq(1,4000,length.out=100),]
colnames(draws)[1] <- "mu"
draws$id<-1:100
postdf <- pmap_df(draws, ~ data_frame(x = seq(-4, 4, length.out = 101), id=..3,
                                      density = dnorm(x, ..1, ..2)))

p2s <- p1 + 
  geom_line(data=postdf, aes(group = id, x = x, y = density),
            linetype=1, color="blue", alpha=0.2) +
  labs(title = "Gaussians with posterior draw parameters", y="density") 
p2s

p2sm <- p2s + 
  geom_segment(data=draws, aes(x=mu, xend=mu, y=0, yend=dnorm(0,0,sigma)),
               linetype=1, color="blue", alpha=0.2) +
  xlab(TeX("y / $\\mu$"))
p2sm

ggplot(draws, aes(x=mu, y=sigma)) +
  geom_point(color="blue") +
  labs(x=TeX("$\\mu$"), y=TeX("$\\sigma$"),
       title = "Draws from the joint posterior distribution")

drawsall <- as.data.frame(fit_1)
colnames(drawsall)[1] <- "mu"
ggplot(drawsall, aes(x=mu, y=sigma)) +
  geom_point(color="blue") +
  labs(x=TeX("$\\mu$"), y=TeX("$\\sigma$"),
       title = "Draws from the joint posterior distribution")




























