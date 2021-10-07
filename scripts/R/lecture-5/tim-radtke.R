
# This script implements the Local-Linear Trend model of the 
# Bayesian Structural Time Series family of models. 
# It closely follows the description available on
# http://www.unofficialgoogledatascience.com/2017/07/fitting-bayesian-structural-time-series.html
# and the tips regarding the Stan implementation available on:
# https://discourse.mc-stan.org/t/bayesian-structural-time-series-modeling/2256/2

# The accompanying blog post is available at:
# https://minimizeregret.com/post/2020/06/07/rediscovering-bayesian-structural-time-series/

# The graphs below are based on the great examples provided at:
# https://mjskay.github.io/tidybayes/

################################################################################

library(dplyr)
library(tidybayes)

library(gganimate)
#library(transformr)
#library(gifski)

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

################################################################################

# We generate a random time series from our model specification

set.seed(4539)

T <- 40

y <- rep(NA, T)
mu_err <- rnorm(T, 0, 1)
delta_err <- rnorm(T, 0, 1)
s_obs <- abs(rnorm(1, 0, 10))
s_slope <- abs(rnorm(1, 0, 0.5))
s_level <- abs(rnorm(1, 0, 0.5))

mu <- rep(NA, T)
delta <- rep(NA, T)

mu[1] <- mu_err[1];
delta[1] <- delta_err[1];
for (t in 2:T) {
  mu[t] <- mu[t-1] + delta[t-1] + s_level * mu_err[t];
  delta[t] <- delta[t-1] + s_slope * delta_err[t];
}

y <- rnorm(T, mu, s_obs)

print(paste0("s_obs = ", round(s_obs, 2)))
print(paste0("s_slope = ", round(s_slope, 2)))
print(paste0("s_level = ", round(s_level, 2)))

plot(y, pch = 19, cex = abs(delta))
lines(mu)

################################################################################

# Use the local-linear trend model defined in `local_linear_trend.stan` to
# estimate the model using Stan on the generated time series.

llt_model <- stan_model("local_linear_trend.stan",
                        model_name = "local_linear_trend")
llt_fit <- sampling(object = llt_model, 
                    data = list(T = T, y = y),
                    chains = 4,
                    iter = 4000,
                    seed = 357,
                    verbose = TRUE,
                    control = list(adapt_delta = 0.95))

print(llt_fit, pars = c("s_obs", "s_level", "s_slope"))
pairs(llt_fit, pars = c("s_obs", "s_level", "s_slope"))

################################################################################

# try different ways of visualizing the fitted model and forecasts

# Posterior distributions of the scale parameters
gather_draws(llt_fit, s_obs, s_level, s_slope) %>%
  ggplot(aes(x = .value, y = .chain)) +
  geom_halfeyeh() +
  facet_wrap(~.variable, scales = "free")

# Boxplot of the trend posterior distribution at each time step
gather_draws(llt_fit, mu[i]) %>%
  ggplot(aes(x = i, y = .value)) +
  geom_boxplot(aes(group = i)) +
  facet_wrap(~.chain, scales = "free")

# Shaded intervals of the trend posterior distribution over time
gather_draws(llt_fit, mu[i]) %>%
  ggplot(aes(x = i, y = .value)) +
  stat_lineribbon(aes(y = .value), .width = c(.99, .95, .8, .5), 
                  color = "#08519C") +
  facet_wrap(~.chain, scales = "free") +
  scale_fill_brewer()

# Shaded intervals of the slope of the linear trend posterior distribution
gather_draws(llt_fit, delta[i]) %>%
  ungroup() %>%
  ggplot(aes(x = i, y = .value)) +
  stat_lineribbon(.width = c(.99, .95, .8, .5), color = "#08519C") +
  facet_wrap(~.chain, scales = "free") +
  scale_fill_brewer() +
  labs(title = "Slope of Linear Trend over Time",
       subtitle = "Positive values imply increasing level, negative values imply decreasing level.")

# Shaded intervals of the trend posterior distribution over time
# at same y-axis scale as posterior predictive distribution below
gather_draws(llt_fit, mu[i]) %>%
  ggplot(aes(x = i, y = .value)) +
  stat_lineribbon(aes(y = .value), 
                  .width = c(.99, .95, .8, .5), 
                  color = "#08519C") +
  facet_wrap(~.chain, scales = "free") +
  scale_fill_brewer() +
  coord_cartesian(ylim = c(-30,50)) +
  labs(x = "Time", y = "mu_t",
       title = "Posterior of Trend over Time")

# derive the median innovation of the slope of the trend for the next plot
s_obs_draws <- spread_draws(llt_fit, s_obs)
s_delta_median <- gather_draws(llt_fit, delta[i]) %>%
  group_by(i) %>%
  summarize(delta = median(.value)) %>%
  mutate(y = y)

# Posterior predictive distribution over time
gather_draws(llt_fit, mu[i]) %>%
  ungroup() %>%
  inner_join(s_obs_draws, by = c(".chain", ".iteration", ".draw")) %>%
  mutate(y = rnorm(n(), mean = .value, sd = s_obs)) %>%
  ggplot(aes(x = i, y = y)) +
  stat_lineribbon(.width = c(.99, .95, .8, .5), color = "#08519C") +
  #geom_line(data = data.frame(i = 1:40, y = y)) +
  geom_point(aes(alpha = abs(delta), size = abs(delta)), 
             data = s_delta_median) +
  facet_wrap(~.chain, scales = "free") +
  scale_fill_brewer() +
  scale_size_continuous(range = c(0.1, 2)) +
  coord_cartesian(ylim = c(-30,50))

################################################################################

# Function that generates forecasts given the posterior samples
# of the fitted model for the next h time steps

forecast_llt <- function(llt_fit, h = 10) {
  
  s_draws <- spread_draws(llt_fit, s_obs, s_level, s_slope)
  all_draws <- gather_draws(llt_fit, mu[i], delta[i]) %>%
    ungroup() %>%
    filter(i == max(i)) %>%
    tidyr::spread(.variable, .value) %>%
    inner_join(s_draws, by = c(".chain", ".iteration", ".draw"))
  all_draws_orig <- all_draws
  
  steps <- max(all_draws$i) + 1:h
  
  y_draws <- expand.grid(i = steps, .draw = unique(all_draws$.draw)) %>%
    inner_join(distinct(all_draws, .chain, .iteration, .draw), by = ".draw")
  y_draws$y <- NA
  
  for(i in 1:h) {
    all_draws <- all_draws %>%
      mutate(delta = delta + rnorm(n(), 0, s_slope)) %>%
      mutate(mu = mu + delta + rnorm(n(), 0, s_level)) %>%
      mutate(y = mu + rnorm(n(), 0, s_obs))
    
    y_draws[y_draws$i == steps[i],]$y <- all_draws$y
  }
  
  return(list(y_draws = y_draws, pars_draws = all_draws))
}

# draw samples from the forecast distribution to then plot them
llt_fc <- forecast_llt(llt_fit)

# Actual historical observations and forecast distribution intervals
ggplot(llt_fc$y_draws, aes(x = i, y = y)) +
  stat_lineribbon(.width = c(.99, .95, .8, .5), color = "#08519C") +
  #geom_line(data = data.frame(i = 1:40, y = y)) +
  geom_point(data = s_delta_median) +
  facet_wrap(~.chain, scales = "free") +
  scale_fill_brewer() +
  scale_size_continuous(range = c(0.1, 2))

# Generate a single possible realization that continues the time series we
# generated at the beginning of the script; we pretend that these are the
# actuals we observed after generating our forecast

set.seed(4739)

h <- 10
mu_test <- mu
delta_test <- delta
mu_err_test <- rnorm(h, 0, 1)
delta_err_test <- rnorm(h, 0, 1)

for (t in T+(1:h)) {
  mu_test[t] <- mu_test[t-1] + delta_test[t-1] + s_level * mu_err_test[t-T];
  delta_test[t] <- delta_test[t-1] + s_slope * delta_err_test[t-T];
}

y_test <- rnorm(h, mu_test[T+(1:h)], s_obs)
data_test <- data.frame(i = T+(1:h), y = y_test)

# plot historical actuals, future observations, as well as the forecast
# to show that the forecast matches the realized observations
ggplot(llt_fc$y_draws, aes(x = i, y = y)) +
  stat_lineribbon(.width = c(.99, .95, .8, .5), color = "#08519C") +
  #geom_line(data = data.frame(i = 1:40, y = y)) +
  geom_point(data = s_delta_median) +
  geom_point(data = data_test) +
  facet_wrap(~.chain, scales = "free") +
  scale_fill_brewer() +
  scale_size_continuous(range = c(0.1, 2))

################################################################################

# Create an animated graph showing different possible future realizations
# of our time series as forecasted by our fitted model, following its
# forecast distribution

set.seed(4729)
ndraws <- 50
sampled_draws <- sample(1:max(llt_fc$y_draws$.draw), ndraws)

p <- llt_fc$y_draws %>%
  filter(.draw %in% sampled_draws) %>%
  ggplot(aes(x = i, y = y)) +
  geom_point(data = s_delta_median) +
  geom_line(aes(group = .draw), color = "#08519C") +
  transition_states(.draw, 0, 1) +
  shadow_mark(past = TRUE, future = FALSE, alpha = 1/20, color = "gray50")

animate(p, nframes = ndraws, fps = 2.5, 
        width = 700, height = 432, res = 100, 
        dev = "png")