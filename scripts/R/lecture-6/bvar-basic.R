# Load the package
library("BVAR")

# Access a subset of the fred_qd dataset
data <- fred_qd[, c("GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS")]
# Transform it to be stationary
data <- fred_transform(data, codes = c(5, 5, 5, 1), lag = 4)

# Estimate using default priors and MH step
x <- bvar(data, lags = 2, n_draw = 40000)

# Check convergence via trace and density plots
plot(x)

# Calculate and store forecasts and impulse responses
predict(x) <- predict(x, horizon = 50)
irf(x) <- irf(x, horizon = 20, identification = TRUE)

# Plot forecasts and impulse responses
plot(predict(x))
plot(irf(x))
