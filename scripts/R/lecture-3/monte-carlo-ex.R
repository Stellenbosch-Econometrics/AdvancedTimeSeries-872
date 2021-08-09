
# Exercise 2: Monte Carlo methods 

# This is the exercise for the second lecture!

# Compute an approximation to π using Monte Carlo. For random number generation use only runif().
# Your hints are as follows:
  
# 1. If U is a bivariate uniform random variable on the unit square (0,1)^2, then the probability that U lies in a subset B of (0,1)^2 is equal to the area of B.
# 2. If U1,…,Un are i.i.d copies of U, then, as n gets larger, the fraction that falls in B converges to the probability of landing in B.
# 3. For a circle, area = π * radius^2. 

## This exercise is a bit more difficult than the previous one, so you might want to take some time to think about it. Draw out your ideas on a piece of paper, that's how I managed to solve this...! You can easily find the answer to this online, but there is no fun in that. 

mc_pi <- function(n){
  count = 0
  for (i in 1:n) {
    # 
    # if () { 
    #  count = count + 1
    }
  }
  area_est <- 
  print(area_est * )
}

