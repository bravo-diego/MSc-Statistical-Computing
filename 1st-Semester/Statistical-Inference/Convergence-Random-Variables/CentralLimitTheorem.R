# Central Limit Theorem

  # The Central Limit Theorem states that if you have a population with mean mu and standard deviation sigma and take sufficiently large random samples from the population with replacement, then the distribution of the sample means will be approximately normally distributed.

# Write the following function. Simulate a sample of size n of a random variable X~Exp(lambda) and calculate Zn:

  # Zn = sqrt(n)(Xn - lambda^{-1})(1 / lambda)

    # repeat this process m times.

# N o t e. Function should take n, m, and lambda as parameters and return a vector with length n containing the sample Zn.

ZFunction <- function(n, m, lambda){ # parameters(sample size, repeat m times, lambda value)
  ZnValues <- numeric(m) # empty vector to store Zn values
  for(i in 1:m){
    sample <- rexp(n, rate = lambda) # random sample of X~Exp(lambda)
    Zn <- (sqrt(n)*(mean(sample) - (1/lambda)))/(1/lambda) 
    ZnValues[i] <- Zn # store Zn values
  }
  return(ZnValues)
}

# Run the function with n = (5, 10, 100, 500, 1000, 10000), m = 1000, and lambda value lambda = 1. Plot Zn samples.

m <- 1000
lambda <- 1

N5 <- ZFunction(5, m, lambda) # n = 5
hist(N5, main = expression("Histogram of " * Z[m] * " with X ~ Exp(" * lambda * ")"), xlab = expression(Z[m]), col = "steelblue", border = "black")
mtext(substitute(paste(italic("n = 5"))), side = 3)

N10 <- ZFunction(10, m, lambda) # n = 10
hist(N10, main = expression("Histogram of " * Z[m] * " with X ~ Exp(" * lambda * ")"), xlab = expression(Z[m]), col = "steelblue", border = "black")
mtext(substitute(paste(italic("n = 10"))), side = 3)

N100 <- ZFunction(100, m, lambda) # n = 100
hist(N100, main = expression("Histogram of " * Z[m] * " with X ~ Exp(" * lambda * ")"), xlab = expression(Z[m]), col = "steelblue", border = "black")
mtext(substitute(paste(italic("n = 100"))), side = 3)

N500 <- ZFunction(500, m, lambda) # n = 500
hist(N500, main = expression("Histogram of " * Z[m] * " with X ~ Exp(" * lambda * ")"), xlab = expression(Z[m]), col = "steelblue", border = "black")
mtext(substitute(paste(italic("n = 500"))), side = 3)

N1000 <- ZFunction(1000, m, lambda) # n = 10
hist(N1000, main = expression("Histogram of " * Z[m] * " with X ~ Exp(" * lambda * ")"), xlab = expression(Z[m]), col = "steelblue", border = "black")
mtext(substitute(paste(italic("n = 1000"))), side = 3)

N10000 <- ZFunction(10000, m, lambda) # n = 10
hist(N10000, main = expression("Histogram of " * Z[m] * " with X ~ Exp(" * lambda * ")"), xlab = expression(Z[m]), col = "steelblue", border = "black")
mtext(substitute(paste(italic("n = 10000"))), side = 3)

# Find both QQ and PP plots for every sample (i.e. n = (5, 10, ...))

  # QQ plot compares the quantiles of a data distribution with the quantiles of a standarized theoretical distribution from a specified family of distributions

qqnorm(N5, main = "QQ Plot") # QQ plot for sample with n = 5
qqline(N5) # on a qq plot the reference line representing a particular theoretical distribution depends on the location and scale parameters of that distribution, having intercept and slope equal to the location and scale parameters
mtext(substitute(paste(italic("n = 5"))), side = 3)

  # PP plot compares the empirical cumulative distribution function of a data set with a specified theoretical cumulative distribution function

probDist <- pnorm(N5)  # probability distribution for Zn values with n = 5
plot(ppoints(m), sort(probDist), main = paste("PP Plot"), 
     xlab = "Observed Probability", ylab = "Expected Probability")
mtext(substitute(paste(italic("n = 5"))), side = 3)
abline(0, 1) # on a pp plot the reference line for any distribution always is the diagonal line y=x

qqnorm(N10, main = "QQ Plot") # QQ plot for sample with n = 10
qqline(N10) # on a qq plot the reference line representing a particular theoretical distribution depends on the location and scale parameters of that distribution, having intercept and slope equal to the location and scale parameters
mtext(substitute(paste(italic("n = 10"))), side = 3)

probDist <- pnorm(N10)  # probability distribution for Zn values with n = 10
plot(ppoints(m), sort(probDist), main = paste("PP Plot"), 
     xlab = "Observed Probability", ylab = "Expected Probability")
mtext(substitute(paste(italic("n = 10"))), side = 3)
abline(0, 1) # on a pp plot the reference line for any distribution always is the diagonal line y=x

qqnorm(N100, main = "QQ Plot") # QQ plot for sample with n = 100
qqline(N100) # on a qq plot the reference line representing a particular theoretical distribution depends on the location and scale parameters of that distribution, having intercept and slope equal to the location and scale parameters
mtext(substitute(paste(italic("n = 100"))), side = 3)

probDist <- pnorm(N100)  # probability distribution for Zn values with n = 100
plot(ppoints(m), sort(probDist), main = paste("PP Plot"), 
     xlab = "Observed Probability", ylab = "Expected Probability")
mtext(substitute(paste(italic("n = 100"))), side = 3)
abline(0, 1) # on a pp plot the reference line for any distribution always is the diagonal line y=x

qqnorm(N500, main = "QQ Plot") # QQ plot for sample with n = 500
qqline(N500) # on a qq plot the reference line representing a particular theoretical distribution depends on the location and scale parameters of that distribution, having intercept and slope equal to the location and scale parameters
mtext(substitute(paste(italic("n = 500"))), side = 3)

probDist <- pnorm(N500)  # probability distribution for Zn values with n = 500
plot(ppoints(m), sort(probDist), main = paste("PP Plot"), 
     xlab = "Observed Probability", ylab = "Expected Probability")
mtext(substitute(paste(italic("n = 500"))), side = 3)
abline(0, 1) # on a pp plot the reference line for any distribution always is the diagonal line y=x

qqnorm(N1000, main = "QQ Plot") # QQ plot for sample with n = 1000
qqline(N1000) # on a qq plot the reference line representing a particular theoretical distribution depends on the location and scale parameters of that distribution, having intercept and slope equal to the location and scale parameters
mtext(substitute(paste(italic("n = 1000"))), side = 3)

probDist <- pnorm(N1000)  # probability distribution for Zn values with n = 1000
plot(ppoints(m), sort(probDist), main = paste("PP Plot"), 
     xlab = "Observed Probability", ylab = "Expected Probability")
mtext(substitute(paste(italic("n = 1000"))), side = 3)
abline(0, 1) # on a pp plot the reference line for any distribution always is the diagonal line y=x

qqnorm(N10000, main = "QQ Plot") # QQ plot for sample with n = 10000
qqline(N10000) # on a qq plot the reference line representing a particular theoretical distribution depends on the location and scale parameters of that distribution, having intercept and slope equal to the location and scale parameters
mtext(substitute(paste(italic("n = 10000"))), side = 3)

probDist <- pnorm(N10000)  # probability distribution for Zn values with n = 10000
plot(ppoints(m), sort(probDist), main = paste("PP Plot")
     xlab = "Observed Probability", ylab = "Expected Probability")
mtext(substitute(paste(italic("n = 10000"))), side = 3)
abline(0, 1) # on a pp plot the reference line for any distribution always is the diagonal line y=x



