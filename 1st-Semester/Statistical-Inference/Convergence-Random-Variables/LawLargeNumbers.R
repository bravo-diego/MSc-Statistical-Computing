# Law of Large Numbers (LLN)

  # Theorem describes the result of performing the same experiment a large number of times; if you repeat an experiment independently a large number of times and average the result, what obtain should be close to the expected value.

# Simulate a sample {X_{1}, ..., X_{n}} of a random variable X~Normal(pi, sqrt(2)) with n = 10^5. Let y_{m} = sum^{m}_{i = 1} x_{i}/m; plot y_{m} as function of m.

mu <- pi # mean
sd <- sqrt(2) # standard deviation
n <- 10^(5) # sample size

sample <- rnorm(n, mu, sd) 
ym <- cumsum(sample) / (1:n) # y_{m} = sum^{m}_{i = 1} x_{i}/m

plot(c(1:n), ym, type = 'l', col = "steelblue3", lwd = 2, 
     xlab = "m", ylab = expression(y[m]),
     main = expression("Function " * y[m] == sum(x[i]/m, i == 1, m) * " with X ~ Normal(" * pi * "," * sqrt(2) * ")")) # plot the y_{m} values

# Repeat this process 100 times and plot y_{m} of every iteration over the same plot.

mu <- pi # mean
sd <- sqrt(2) # standard deviation
n <- 10^(5) # sample size

iterations <- 100
ymatrix <- matrix(0, nrow = n, ncol = iterations)

for(i in 1:iterations){
  sample <- rnorm(n, mu, sd)
  ym <- cumsum(sample) / (1:n) # y_{m} = sum^{m}_{i = 1} x_{i}/m
  ymatrix[, i] <- ym
}

matplot(c(1:n), ymatrix, type = 'l', col = "steelblue3", lwd = 2,
        xlab = "m", ylab = expression(y[m]),
        main = expression("Function " * y[m] == sum(x[i]/m, i == 1, m) * " with X ~ Normal(" * pi * "," * sqrt(2) * ")"))
mtext(substitute(paste(italic("simulation repeated 100 times"))), side = 3)

# Do the same with X~Cauchy(pi, sqrt(2)).

  # The Cauchy distribution is the student's t distribution with one degree of freedom. The Cauchy distribution does not have a well defined mean or variance. 

location <- pi # location parameter
scale <- sqrt(2) # scale parameter
n <- 10^(5) # sample size

sample <- rcauchy(n, location, scale)
ym <- cumsum(sample) / (1:n) # y_{m} = sum^{m}_{i = 1} x_{i}/m

plot(c(1:n), ym, type = 'l', col = "orange", lwd = 2, 
     xlab = "m", ylab = expression(y[m]),
     main = expression("Function " * y[m] == sum(x[i]/m, i == 1, m) * " with X ~ Cauchy(" * pi * "," * sqrt(2) * ")")) # plot the y_{m} values

location <- pi # location parameter
scale <- sqrt(2) # scale parameter
n <- 10^(5) # sample size

iterations <- 100
ymatrix <- matrix(0, nrow = n, ncol = iterations)

for(i in 1:iterations){
  sample <- rcauchy(n, location, scale)
  ym <- cumsum(sample) / (1:n) # y_{m} = sum^{m}_{i = 1} x_{i}/m
  ymatrix[, i] <- ym
}

matplot(c(1:n), ymatrix, type = 'l', col = "orange", lwd = 2,
        xlab = "m", ylab = expression(y[m]),
        main = expression("Function " * y[m] == sum(x[i]/m, i == 1, m) * " with X ~ Cauchy(" * pi * "," * sqrt(2) * ")"))
mtext(substitute(paste(italic("simulation repeated 100 times"))), side = 3)
