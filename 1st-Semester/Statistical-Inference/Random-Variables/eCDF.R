# Given the following data set: [23.37, 21.87, 24.41, 21.27, 23.33, 15.20, 24.21, 27.52, 15.48, 27.19, 25.05, 20.40, 21.05, 28.83, 22.90, 18.00, 17.55, 25.92, 23.64, 28.96, 23.02, 17.32, 30.74, 26.73, 17.22, 22.81, 20.78, 23.71, 21.60, 22.37]

  # Calculate the empirical distribution function associated with the data points and plot the empirical distribution function. 

sample_points <- c(23.37, 21.87, 24.41, 21.27, 23.33, 15.20, 24.21, 27.52, 15.48, 27.19, 
                   25.05, 20.40, 21.05, 28.83, 22.90, 18.00, 17.55, 25.92, 23.64, 28.96, 
                   23.02, 17.32, 30.74, 26.73, 17.22, 22.81, 20.78, 23.71, 21.60, 22.37) # data points

n <- length(sample_points) # n = 30
sample_points <- sort(sample_points) # sort data points; Fn(Xi) = i/n
edf <- cumsum(rep(1/n, n))

plot(sample_points, edf, type = 's', main = 'Empirical Distribution', xlab = 'x', ylab = 'Fn(x)')

    # Adding confidence bands based on the Kolmogorov Smirnov test (alpha = 0.01, 0.05). 

ks_critical_05 <- 0.24170 # critical value D_{n, a} with alpha = 0.05

upper_bound <- numeric(n)
lower_bound <- numeric(n)

for (i in 1:n){
  upper_bound[i] <- pmin(edf[i] + ks_critical_05, 1) # upper limit
  lower_bound[i] <- pmax(edf[i] - ks_critical_05, 0) # lower limit
}

plot(sample_points, edf, type = 's', main = 'Empirical Distribution', xlab = 'x', ylab = 'Fn(x)')
mtext(substitute(paste(italic("Critical value D_{n, a} with alpha = 0.05"))), side = 3)
lines(sample_points, upper_bound, col = 'brown2')
lines(sample_points, lower_bound, col = 'brown2')

ks_critical_01 <- 0.28988 # critical value D_{n, a} with alpha = 0.01

upper_bound <- numeric(n)
lower_bound <- numeric(n)

for (i in 1:n){
  upper_bound[i] <- pmin(edf[i] + ks_critical_01, 1) # upper limit
  lower_bound[i] <- pmax(edf[i] - ks_critical_01, 0) # lower limit
}

plot(sample_points, edf, type = 's', main = 'Empirical Distribution', xlab = 'x', ylab = 'Fn(x)')
mtext(substitute(paste(italic("Critical value D_{n, a} with alpha = 0.01"))), side = 3)
lines(sample_points, upper_bound, col = 'brown2')
lines(sample_points, lower_bound, col = 'brown2')

# Understanding QQ plots - Graphical tool to help us assess if a set of data plausibly came from some theoretical distribution such as a normal or exponential (i.e. if we run a statistical analysis that assumes our residuals are normally distributed, we can use a normal QQ polot to check that assumption).

  # Plot the Normal Q-Q plot associated with the data points. N o t e: adjust the line sx + m in order to compare the 

Pi <- cumsum(rep(1/(n+1), n)) # percentiles
Zp <- qnorm(Pi) # normalize percentiles

s_standard_deviation <- sd(sample_points) # sample standard deviation
s_mean <- mean(sample_points) # sample mean

x <- ((sample_points-s_mean) / s_standard_deviation)

plot(Zp, sample_points, type = 'o', main = 'Normal Q-Q Plot', xlab = 'Theoretical Quantiles', ylab = 'Sample Quantiles')
par(new=TRUE)
plot(x, sample_points, type='l', xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col = 'brown2')

# Add confidence bands with alpha value alpha = 0.01, 0.05, based on the Kolmogorov Smirnov test. 

  # The values D_{n,a} were obtained from real-statistics.com (URL: https://real-statistics.com/statistics-tables/kolmogorov-smirnov-table/)

ks_critical_05 <- 0.24170 # critical value D_{n, a}

upper <- Pi - ks_critical_05
lower <- Pi + ks_critical_05

upper_bound <- qnorm(upper)
lower_bound <- qnorm(lower)

upper_expectedvalues <- upper_bound * s_standard_deviation + s_mean
lower_expectedvalues <- lower_bound * s_standard_deviation + s_mean

plot(Zp, sample_points, type = 'o', main = 'Normal Q-Q Plot', xlab = 'Theoretical Quantiles', ylab = 'Sample Quantiles')
mtext(substitute(paste(italic("Critical value D_{n, a} with alpha = 0.05"))), side = 3)
par(new=TRUE)
plot(x, sample_points, type='l', xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col = 'brown2')
lines(Zp, upper_expectedvalues, col = 'brown2', lty = 2, lwd = 2)
lines(Zp, lower_expectedvalues, col = 'brown2', lty = 2, lwd = 2)

ks_critical_01 <- 0.28988 # critical value D_{n, a}

upper <- Pi - ks_critical_01
lower <- Pi + ks_critical_01

upper_bound <- qnorm(upper)
lower_bound <- qnorm(lower)

upper_expectedvalues <- upper_bound * s_standard_deviation + s_mean
lower_expectedvalues <- lower_bound * s_standard_deviation + s_mean

plot(Zp, sample_points, type = 'o', main = 'Normal Q-Q Plot', xlab = 'Theoretical Quantiles', ylab = 'Sample Quantiles')
mtext(substitute(paste(italic("Critical value D_{n, a} with alpha = 0.01"))), side = 3)
par(new=TRUE)
plot(x, sample_points, type='l', xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col = 'brown2')
lines(Zp, upper_expectedvalues, col = 'brown2', lty = 2, lwd = 2)
lines(Zp, lower_expectedvalues, col = 'brown2', lty = 2, lwd = 2)

# Normal Probability Plot 

  # The Normal probability plot is a graphical technique for assessing whether or not a data set is approximately normally distributed. The data are plotted against a theoretical normal distribution in such a way that the points should form an approximate straight line.
  
    # In a normal probability plot the sorted data are plotted versus values selected to make the resulting image look close to a straight line if data are approximately normally distributed. 
  
normal_plot <- function(sample_data){
  sorted_data <- sort(sample_data) # ordered observations
  n <- length(sample_data) 
  
  expected_percentiles <- (1:n - 0.5)/n # probability levels - Applied Multivariate Statistical Analysis (Richard A. Johnson)
  theoretical_quantiles <- qnorm(expected_percentiles)  # standard normal quantiles 

  plot(theoretical_quantiles, sorted_data, main = 'Normal Probability Plot', xlab = 'Normal N(0, 1) Order Statistics Medians', ylab = 'Ordered Response')
  par(new=TRUE)
  plot(theoretical_quantiles, theoretical_quantiles, type='l', xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col = 'steelblue')
  
}

normal_plot(sample_points)
