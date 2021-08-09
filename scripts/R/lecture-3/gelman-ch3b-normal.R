
# Normal model with unknown mean and variance (Gelman pg 64)

library(ggplot2)
theme_set(theme_minimal())
library(grid)
library(gridExtra)
library(tidyr)

set.seed(2141) 

# We need to generate some data (this is arbitrary)
y <- c(93, 112, 122, 135, 122, 150, 118, 90, 124, 114)

# Generate the sufficient statistics for this data
n <- length(y)
s2 <- var(y)
my <- mean(y)

# We can factorise the join posterior and sample from the joint posterior using this factorisation
## Two components in this factorisation

# nu is the degrees of freedom while s2 is the scaling parameter
# In our construction we are ignoring the normalising constant, which means this is not a probability distribution -- see the wikipedia entry https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution

rsinvchisq  <- function(n, nu, s2, ...) nu*s2 / rchisq(n, nu, ...) # nu*s2 is the scaling component multiplied by the inverse chi-squared distribution -- see Albert page 64. See also page 583 in Gelman.

# This is a helper function for the exact marginal density of sigma. Will be used later in the process. 
dsinvchisq <- function(x, nu, s2){
  exp(log(nu/2)*nu/2 - lgamma(nu/2) + log(s2)/2*nu - log(x)*(nu/2+1) - (nu*s2/2)/x)
}

# Sample 1000 random numbers from the p(sigma2 | y)
ns <- 1000
sigma2 <- rsinvchisq(ns, n-1, s2)

# Next we sample from p(mu | sigma2, y) -- not exactly sure where this comes from in the book (see Section 2.5)
mu <- my + sqrt(sigma2/n)*rnorm(length(sigma2)) # length(sigma) tells us how many variables to generate. In this case the defaults for rnorm are used. Shifted by ybar and multiplied by standard deviation. 

# Potentially another way to do this would have been... (This makes more intuitive sense)
mu1 <- rnorm(length(sigma2), mean = my, sd = sqrt(sigma2)/sqrt(n))

# Create a variable sigma
sigma <- sqrt(sigma2)

# For mu, sigma compute the density in a grid (ranges of the grid are specified)
t1l <- c(90, 150)
t2l <- c(10, 60)
t1 <- seq(t1l[1], t1l[2], length.out = ns)
t2 <- seq(t2l[1], t2l[2], length.out = ns)

# We can also compute the exact marginal density (analytical) of mu
## See page 21 for the discussion on the transformation of a variable
## Multiply by 1./sqrt(s2/n) since z = (x - mean(y)) / sqrt(s2/n)

pm <- dt((t1-my) / sqrt(s2/n), n-1) / sqrt(s2/n) ## see page 66 for the t-density representation.

# Next we estimate the marginal density of mu using samples and Gaussian kernel approximation
pmk <- density(mu, adjust = 2, n = ns, from = t1l[1], to = t1l[2])$y # How does this work?

# Compute the exact marginal density of sigma
## Take note of the transformation multiplication by 2*t2 since z = t2^2
ps <- dsinvchisq(t2^2, n-1, s2) * 2*t2

# Estimate the marginal density of sigma using samples and Gaussian kernel approximation 
psk <- density(sigma, n = ns, from = t2l[1], to = t2l[2])$y

# Evaluate the join density in a grid. We will combine grid points into another data frame with all pairwise combinations (need to go through this in more detail to fully understand it)
dfj <- data.frame(t1 = rep(t1, each = length(t2)),
                  t2 = rep(t2, length(t1)))

# We have constructed a data frame where each t1 value is repeated 1000 times to make the first column of the data frame. Second column is the repetition of t2 vector of values (one vector at a time) a thousand times. Best way to understand this is to do some data exploration. 

dfj$z <- dsinvchisq(dfj$t2^2, n-1, s2) * 2*dfj$t2 * dnorm(dfj$t1, my, dfj$t2/sqrt(n))

# Add some breaks for plotting the contours
cl <- seq(1e-5, max(dfj$z), length.out = 6)

#### Visualise the joint and marginal densities ####

# First we have a plot of marginal density of mu

dfm <- data.frame(t1, Exact = pm, Empirical = pmk) %>% gather(grp, p, -t1)
margmu <- ggplot(dfm) +
  geom_line(aes(t1, p, color = grp)) +
  coord_cartesian(xlim = t1l) +
  labs(title = 'Marginal of mu', x = '', y = '') +
  scale_y_continuous(breaks = NULL) +
  theme(legend.background = element_blank(),
        legend.position = c(0.75, 0.8),
        legend.title = element_blank())
margmu

# Second we have a plot of marginal density of sigma

dfs <- data.frame(t2, Exact = ps, Empirical = psk) %>% gather(grp, p, -t2)
margsig <- ggplot(dfs) +
  geom_line(aes(t2, p, color = grp)) +
  coord_cartesian(xlim = t2l) +
  coord_flip() +
  labs(title = 'Marginal of sigma', x = '', y = '') +
  scale_y_continuous(breaks = NULL) +
  theme(legend.background = element_blank(),
        legend.position = c(0.75, 0.8),
        legend.title = element_blank())
margsig
# Create a plot of the joint density

joint1labs <- c('Samples','Exact contour')
joint1 <- ggplot() +
  geom_point(data = data.frame(mu,sigma), aes(mu, sigma, col = '1'), size = 0.1) +
  geom_contour(data = dfj, aes(t1, t2, z = z, col = '2'), breaks = cl) +
  coord_cartesian(xlim = t1l,ylim = t2l) +
  labs(title = 'Joint posterior', x = '', y = '') +
  scale_y_continuous(labels = NULL) +
  scale_x_continuous(labels = NULL) +
  scale_color_manual(values=c('blue', 'black'), labels = joint1labs) +
  guides(color = guide_legend(nrow  = 1, override.aes = list(
    shape = c(16, NA), linetype = c(0, 1), size = c(2, 1)))) +
  theme(legend.background = element_blank(),
        legend.position = c(0.5, 0.9),
        legend.title = element_blank())

joint1

# Combine the plots
bp <- grid.rect(gp = gpar(col = 'white'))
ga1<-grid.arrange(joint1, margsig, margmu, bp, nrow = 2)
ga1

#' ### Demo 3.2 Visualise factored distribution
#' Visualise factored sampling and the corresponding
#' marginal and conditional densities.
#' 

#' Create another plot of the joint posterior
# data frame for the conditional of mu and marginal of sigma
dfc <- data.frame(mu = t1, marg = rep(sigma[1], length(t1)),
                  cond = sigma[1] + dnorm(t1 ,my, sqrt(sigma2[1]/n)) * 100) %>%
  gather(grp, p, marg, cond)
# legend labels for the following plot
joint2labs <- c('Exact contour plot', 'Sample from joint post.',
                'Cond. distribution of mu', 'Sample from the marg. of sigma')
joint2 <- ggplot() +
  geom_contour(data = dfj, aes(t1, t2, z = z, col = '1'), breaks = cl) +
  geom_point(data = data.frame(m = mu[1], s = sigma[1]), aes(m , s, color = '2')) +
  geom_line(data = dfc, aes(mu, p, color = grp), linetype = 'dashed') +
  coord_cartesian(xlim = t1l,ylim = t2l) +
  labs(title = 'Joint posterior', x = '', y = '') +
  scale_x_continuous(labels = NULL) +
  scale_color_manual(values=c('black', 'red','darkgreen','black'), labels = joint2labs) +
  guides(color = guide_legend(nrow  = 2, override.aes = list(
    shape = c(NA, 16, NA, NA), linetype = c(1, 0, 1, 1)))) +
  theme(legend.background = element_blank(),
        legend.position = c(0.5, 0.85),
        legend.title = element_blank())
joint2

#' Create another plot of the marginal density of sigma
margsig2 <- ggplot(data = data.frame(t2, ps)) +
  geom_line(aes(t2, ps), color = 'blue') +
  coord_cartesian(xlim = t2l) +
  coord_flip() +
  labs(title = 'Marginal of sigma', x = '', y = '') +
  scale_y_continuous(labels = NULL, breaks=NULL)
margsig2

#' Combine the plots
grid.arrange(joint2, margsig2, ncol = 2)

#' ### Demo 3.3 Visualise the marginal distribution of mu
#' Visualise the marginal distribution of mu as a mixture of normals.
#' 

#' Calculate conditional pdfs for each sample
condpdfs <- sapply(t1, function(x) dnorm(x, my, sqrt(sigma2/n)))

#' Create a plot of some of them
# data frame of 25 first samples
dfm25 <- data.frame(t1, t(condpdfs[1:25,])) %>% gather(grp, p, -t1)
dfmean <- data.frame(t1, p=colMeans(condpdfs))
condmu <- ggplot(data = dfm25) +
  geom_line(aes(t1, p, group = grp), linetype = 'dashed') +
  labs(title = 'Cond distr of mu for 25 draws', y = '', x = '') +
  scale_y_continuous(breaks = NULL, limits = c(0, 0.1))
condmu

condmu +
  geom_line(data = dfmean, aes(t1, p), color = 'orange', size=2)

#' create a plot of their mean
dfsam <- data.frame(t1, colMeans(condpdfs), pm) %>% gather(grp,p,-t1)
# labels
mulabs <- c('avg of sampled conds', 'exact marginal of mu')
meanmu <- ggplot(data = dfsam) +
  geom_line(aes(t1, p, size = grp, color = grp)) +
  labs(y = '', x = '', title = 'Cond. distr of mu') +
  scale_y_continuous(breaks = NULL, limits = c(0, 0.1)) +
  scale_size_manual(values = c(2, 0.8), labels = mulabs) +
  scale_color_manual(values = c('orange', 'black'), labels = mulabs) +
  theme(legend.position = c(0.8, 0.8),
        legend.background = element_blank(),
        legend.title = element_blank())
meanmu

#' Combine the plots
grid.arrange(condmu, meanmu, ncol = 1)

































