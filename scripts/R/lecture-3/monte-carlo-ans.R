
## Exercise 2: Monte Carlo methods

# Compute an approximation to π using Monte Carlo. For random number generation use only runif().
# Your hints are as follows:
  
# 1. If U is a bivariate uniform random variable on the unit square (0,1)^2, then the probability that U lies in a subset B of (0,1)^2 is equal to the area of B.
# 2. If U1,…,Un are i.i.d copies of U, then, as n gets larger, the fraction that falls in B converges to the probability of landing in B.
# 3. For a circle, area = π * radius^2. 

## This exercise is a bit more difficult than the previous one, so you might want to take some time to think about it. Draw out your ideas on a piece of paper, that's how I managed to solve this...! You can easily find the answer to this online, but there is no fun in that. 

# For loops are great for pedagogy!

mc_pi <- function(n){
  count <- 0 
  for (i in 1:n) {
    u <- runif(2)
    if (u[1]^2 + u[2]^2 < 1) {
      count <- count + 1
    } 
  }
  area_est <- count / n
  print(area_est * 4)
}

# There are some cool examples out there on how to do this. People come up with all different types of methods do this. Here are some of the ones that I found interesting. 

# The following code is vectorised, so we don't have to use the for loop. 

mc_piv <- function(n) {
  x <- runif(n, 0, 1)
  y <- runif(n, 0, 1)
  is_inside <- (x^2 + y^2) < 1
  pi_estimate <- 4 * sum(is_inside) / n
  print(pi_estimate)
}

# Let's try to plot some of this to get a better intuition of what is happening. 

n <- 10000 # Change the value of n to see what is happening
x <- runif(n,0,1)
y <- runif(n,0,1)
d <- (x^2+y^2<1)
plot(x,y,col=d+1,pch=20)


# We can time the code to see which one gives the best result. Which one do you expect to work better?
system.time(a <- mc_pi(10000000))
system.time(a <- mc_piv(10000000))

# Here is another version of the answer... a bit different from the ones we stated above. However, it follows the same idea. 

library(dplyr)

# Short version
mc_pi_short <- function(n){
  count = 0 
  for (i in 1:n) {
    u <- runif(2)
    d <- sqrt((u[1]- 0.5)^2 + (u[2]-0.5)^2) 
    if (d<0.5) {
      count = count + 1
    } 
  }
  area_est <- count / n
  print(area_est * 4)
}

mc_piv_short <- function(n) {
  x <- runif(n, 0, 1)
  y <- runif(n, 0, 1)
  is_inside <- (sqrt((x-0.5)^2 + (y-0.5)^2) < 0.5)
  pi_estimate <- 4 * sum(is_inside) / n
  print(pi_estimate)
}

n <- 10000 # Change the value of n to see what is happening
x <- runif(n,0,1)
y <- runif(n,0,1)
d <- (sqrt((x-0.5)^2 + (y-0.5)^2) < 0.5)
plot(x,y,col=d+1,pch=20)


# Longer version

# Initialise the components

n <- 1000
points <- tibble(x=numeric(n),y=numeric(n))
pi_est <- numeric(n)
inner <-0
outer <-0

# Construct the circle from which we will determine the value of pi

circle <- tibble(x=1:360,y=1:360)

for(i in 1:360){
  circle$x[i] <- 0.5 + cos(i/180*pi)*0.5
  circle$y[i] <- 0.5 + sin(i/180*pi)*0.5
}

for(i in 1:n){
  
  # Draw a new point at random
  points$x[i] <-runif(1)
  points$y[i] <-runif(1)
  
  # Check if the point is inside the circle
  if( (points$x[i]-0.5)^2 + (points$y[i]-0.5)^2 > 0.25 )
  {
    outer=outer+1
  }else
  {
    inner=inner+1
  }
}

current_pi<-(inner/(outer+inner))/(0.25)
pi_est[i]= current_pi

plot(points$x[1:i],points$y[1:i],
     col="red",
     main=c('Estimate of pi: ',formatC(current_pi, digits=4, format="g", flag="#")),
     cex=0.5,pch=19,ylab='',xlab='',xlim=c(0,1),ylim=c(0,1))
lines(circle$x,circle$y,lw=4,col="blue")


# Let us continue with Monte Carlo methods using other examples

# We will see that the empirical distribution of iid draws from a distribution will approach that distribution as we increase the number of draws. Look at the code below (adapted from Hoff Ch4) for some intuition. 

a = 68
b = 45

std.gamma = data.frame(theta = seq(0, 3, by = 0.01), p = dgamma(seq(0, 3, by = 0.01), a, b))
mc10 = data.frame(theta = rgamma(10, a, b), type = '10 samples')
mc100 = data.frame(theta = rgamma(100, a, b), type = '100 samples')
mc1000 = data.frame(theta = rgamma(1000, a, b), type = '1000 samples')

mcs = rbind(mc10, mc100, mc1000)

ggplot(mcs, aes(x = theta, y = ..density..)) +
  geom_histogram(bins = 10, fill = NA, color = 'black') +
  stat_density(fill = NA, color = 'black') +
  scale_x_continuous(limits = c(1, 2)) +
  geom_line(data = std.gamma, mapping = aes(x = theta, y = p), lty = 2, color = 'blue') +
  ylab('density') +
  facet_grid(. ~ type)

# This shows the convergence of the Monte Carlo estimates to the correct values graphically. This is based on cumulative estimates from a sequence of 1000 samples, from the Gamma(68,45) distribution. 

# From this we can calculate the sample mean of the Monte Carlo samples. From the central limit theorem, the mean is approximately normally distributed with a specific mean and standard deviation that can serve as unbiased estimates for the population mean and standard deviation.  

# The code below shows how quickly our estimates converge to the true mean (true mean is known in this case since are using the Gamma distribution)

library(reshape)

cstats = melt(data.frame(
  n = seq_along(mc1000$theta),
  'Expectation' = cumsum(mc1000$theta) / seq_along(mc1000$theta)
), id.vars = 'n')

realstats = data.frame(
  stat = 68 / 45,
  variable = 'Expectation'
)

ggplot(cstats, aes(x = n, y = value)) +
  facet_wrap(~ variable, scales = 'free') +
  geom_line() +
  ylab('Estimate') + xlab('Number of samples') +
  geom_hline(data = realstats, mapping = aes(yintercept = stat), lty = 2)















