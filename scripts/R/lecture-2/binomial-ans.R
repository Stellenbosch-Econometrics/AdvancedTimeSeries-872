
## Exercise 1: Write a function that generates a draw from a random variable that has a binomial distribution.

# The binomial random variable Y ~ Bin(n, p) represents the number of successes in n binary trials where each trial succeeds with probability p. Use the runif() command to write a function called binomial_rv that generates one draw of Y.

# Hint: If U is uniform on (0, 1) and p ∈ (0, 1) then the expression U < p evaluates to true with probability p

binomial_rv <- function(n, p) {
  # Write the function body here
  y <- c(1:n)
  count <- 0
  for (i in seq_along(y)) {
    if (runif(n)[i] < p) {
      count <- count + 1
    }
  }
  # return(count)
}

binomial_rv_new <- function(n, p) {
  # Write the function body here
  count <- 0
  for (i in 1:n) {
    if (runif(n)[i] < p) {
      count <- count + 1
    }
  }
  # return(count)
}

system.time(a <- binomial_rv(10000, 0.5))
system.time(a <- binomial_rv_new(10000, 0.5))

# Seq_along does not make a difference with respect to the time. Maybe it has to do with the memory allocation?
# If you have zero length vector than seq_along is the better option. Otherwise you might get error messages for some cases where the end value is zero. 


# Source for this example (and many more) is https://julia.quantecon.org/getting_started_julia/julia_by_example.html

library(ggplot2)
library(dplyr)

## Below is some extra stuff that I was playing around with (not the best coding style, mostly done for illustration).
# Here we have x draws from Y
x <- 1:10
b <- numeric(length(x))
for (i in seq_along(x)) {
  b[i] <- binomial_rv(1000, 0.5)
}

hist(b) # Provides a simple histogram of the output
ggplot(mapping = aes(b)) + geom_density() # Get a nicer picture using ggplot. 

# Let's see if we can do something interesting, like look at how the central limit theorem works -- CLT is awesome. If you don't know much about it, please read the Wikipedia entry.
z <- 1:10
e <- numeric(length(x))
d <- numeric(length(z))
for (i in seq_along(z)){
  for (j in seq_along(x)){
    e[j] <- binomial_rv(100, 0.2)
  }
  d[i] <- sum(e)  
}

# Generally if you want to speed up the code, one approach is to consider vectorisation, which calls functions in C rather than R code. However, in this case our present iteration depends on results from past iterations, so this is not possible. A good resource for best coding practice in R, is a book called R Inferno, which can be found at https://www.burns-stat.com/pages/Tutor/R_inferno.pdf

# Important to remember that apply methods are not significantly faster, they are simply loops in disguise!! Look at readings on lapply, sapply, tapply, etc. for more information. 

# One possible way to speed this up would perhaps be to use Rcpp, which is basically rewriting key functions in C++. Don't worry, you don't need to learn another language. This will only happen very rarely. 


CLT <- function(x, z){
  e <- rep(0,x)
  d <- rep(0,z)
  for (i in 1:z){
    for (j in 1:x){
      e[j] <- binomial_rv(10, 0.5)
    }
    d[i] <- sum(e)
  }
  return(d)
}

result_clt <- CLT(10, 10000)
ggplot(mapping = aes(result_clt)) + geom_density() 

# Next is an example from the Hoff textbook of a case where Binomial distribution is used in the estimation of the probability of a rare event. In this case, we are going to look at the prevalence of a disease 

# p will be the fraction of infected people in the population
# Take samples from n people with the disease. 

# The sampling model is then given by Y | p ~ Bin(n, p)

# Each individual has an independent p% chance of having the disease. Suppose there are n = 40 individuals

d = data.frame(
  y = 0:100,
  p = factor(rep(c(0.05, 0.15, 0.30), each = 101)),
  probability = c(dbinom(0:100, 100, 0.05), dbinom(0:100, 100, 0.15), dbinom(0:100, 100, 0.3))
)
ggplot(d, aes(x = y, y = probability, fill = p)) +
  geom_bar(stat = "identity", position = "dodge") + geom_density(stat = "identity", position = "dodge", alpha = 0.3)

# Remember this is our sampling model, what we think will happen to the data for different values of p. 

# We can now think about our prior, the likelihood and then calculating the posterior. For computational convenience we will choose a prior that has a Beta distribution (conjugate prior). See the lecture slides. 

d = data.frame(
  p = seq(0, 1, by = 0.001),
  distribution = rep(c("prior", "posterior"), each = 1001),
  density = c(dbeta(seq(0, 1, by = 0.001), 2, 20), dbeta(seq(0, 1, by = 0.001), 2, 40))
)
ggplot(d, aes(x = p, y = density, color = distribution)) +
  geom_line()

# We see that the posterior in this case is more tightly peaked at near-zero values than our prior guess. 

# On to the next issue, let us compute a posterior for the binomial distribution using grid approximation

# Grid approximation of posterior

# Define our grid
p_grid <- seq(from = 0, to = 1, length.out = 1e4) 

# Define the prior
prior <- rep(1, 1e4) 
plot(prior)

# Compute the likelihood at each value of the grid
likelihood <- dbinom(6, size = 9, prob = p_grid) 

# Given the exercise that we did what do you expect to see here?
plot(likelihood)

# Compute the product of the likelihood and prior
unstd.posterior <- likelihood * prior 
plot(unstd.posterior)

# Standardise the posterior
posterior <- unstd.posterior  / sum(unstd.posterior)

# What difference did the last command make? Why did we do this?

plot( p_grid , posterior , type="b" ,
      ylab="posterior probability" )
mtext( "10000 points" )

# Now that we have a posterior, we can take samples from it! We will be doing this a lot in the computational part of the course (next week)

p_grid <- seq(from = 0, to = 1, length.out = 1e4) # Grid with size 100
samples <- sample(p_grid, prob = posterior, size = 1e4, replace = TRUE)
samples <- unlist(samples) # Need to do this for some obscure reason. Answer found on stackoverflow.
p_grid <- unlist(p_grid)
ggplot(mapping = aes(p_grid, samples)) + geom_point(colour = "black", alpha = 5/10)

# What do you expect to get in terms of a density plot once you take samples?
ggplot(mapping = aes(samples)) + geom_density(fill = "black", alpha = 5/10)

# We will see what sampling from the posterior is useful for in future lectures. 












