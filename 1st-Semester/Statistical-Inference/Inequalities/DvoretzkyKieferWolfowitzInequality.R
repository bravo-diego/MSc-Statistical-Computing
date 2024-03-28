# Given the empirical distribution function associated with the data points, calculate and plot the confidence interval using the Dvoretzky-Kiefer-Wolfowitz inequality. 

sample_points <- c(23.37, 21.87, 24.41, 21.27, 23.33, 15.20, 24.21, 27.52, 15.48, 27.19, 
                   25.05, 20.40, 21.05, 28.83, 22.90, 18.00, 17.55, 25.92, 23.64, 28.96, 
                   23.02, 17.32, 30.74, 26.73, 17.22, 22.81, 20.78, 23.71, 21.60, 22.37) # data points

confidence_interval <- function(data_points, alpha){ # confidence interval based on Dvoretzky-Kiefer-Wolfowitz inequality
  n <- length(data_points) # n value
  data_points <- sort(data_points)
  
  empirical_distribution <- cumsum(rep(1/n, n)) # empirical distribution function
  
  epsilon <- sqrt(( 1/ (2*n)) * log(2 / alpha)) # epsilon value 
  
  lower_bound <- numeric(n) # empty vector 
  upper_bound <- numeric(n) # empty vector 
  
  for (i in 1:n){
    lower_bound[i] <- pmax(empirical_distribution[i] - epsilon, 0) # lower bound L(x) 
    upper_bound[i] <- pmin(empirical_distribution[i] + epsilon, 1) # upper bound U(x) 
  }
  
  plot(data_points, empirical_distribution, type = 's', main = 'Empirical Distribution', xlab = 'x', ylab = 'Fn(x)')
  mtext(expression(paste("Confidence Interval based on Dvoretzky-Kiefer-Wolfowitz inequality (", alpha, " = 0.05)")), side = 3, cex = 0.9)
  lines(data_points, upper_bound, col = 'firebrick2', lty = 2)
  lines(data_points, lower_bound, col = 'firebrick2', lty = 2)
  
}

confidence_interval(sample_points, 0.05) # confidence interval whit alpha = 0.05

confidence_interval(sample_points, 0.01) # confidence interval whit alpha = 0.01