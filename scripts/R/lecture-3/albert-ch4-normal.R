

## R-code from Albert for the normal distribution with 2 unknown parameters

library(LearnBayes)

data(marathontimes)
attach(marathontimes)
d <- mycontour(normchi2post, c(220, 330, 500, 9000), time, xlab = "mean", ylab = "variance")

# Can now summarise this posterior distribution by simulation
## Can simulate a value of (mu, sigma^2) from from the joint posterior

# Simulate value for sigma^2 from S times inverse Chi-square distribution (scaled)
## Sample of 1000 will be used here

S <- sum((time - mean(time))^2)
n <- length(time)
sigma2 <- S/rchisq(1000, n-1) # 

# Simulate draws of the mean using the function rnorm (normal distribution)

mu = rnorm(1000, mean = mean(time), sd = sqrt(sigma2)/sqrt(n))

# Simulation algorithm is now implemented and simulated sample values of (mu, sigma^2) are displayed

points(mu, sigma2)
