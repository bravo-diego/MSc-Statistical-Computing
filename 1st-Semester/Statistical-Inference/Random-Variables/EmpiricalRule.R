# Understanding and Visualizing the Empirical Rule 

  # Is a shorthand used to remember the percentage of values that lie within an interval estimate in a normal distribution: 68%, 95% and 99.7%, of the values lie within one, two, and three standard deviation of the mean, respectively.

# For each distribution listed below, calculate the probabilities contained within the intervals (mu - ksigma, mu + ksigma) for k = 1, 2, 3; and simulate n = 1000 sample points from each distribution and calculate the sample mean and variance, calculate the proportion of sample points that fall within the intervals (mu - ksigma, mu + ksigma) for k = 1, 2, 3. Compare both tables. 

n <- 1000 
probabilities <- c() # empty vector to store probabilities
proportions <- c() # empty vector to store proportions 

# X~Unif(a = -3, b = 3)

a <- -3
b <- 3
mean <- (b + a)/2  
standard_deviation <- sqrt((((a - b)**2)/12))

for(k in 1:3){
  p <- punif(mean + (k*standard_deviation), min = -3, max = 3) - punif(mean - (k*standard_deviation), min = -3, max = 3)
  probabilities <- append(probabilities, p)
}

sample_points <- runif(n, min = a, max = b)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- punif(mean_sample + (k*standard_deviation_sample), min = -3, max = 3) - punif(mean_sample - (k*standard_deviation_sample), min = -3, max = 3)
  proportions <- append(proportions, p)
}

# X~Normal(0, 1)

mean <- 0
standard_deviation <- 1

for(k in 1:3){
  p <- pnorm(mean + (k*standard_deviation), mean, standard_deviation) - pnorm(mean - (k*standard_deviation), mean, standard_deviation)
  probabilities <- append(probabilities, p)
}

sample_points <- rnorm(n, mean, standard_deviation)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- pnorm(mean_sample + (k*standard_deviation_sample), mean_sample, standard_deviation_sample) - pnorm(mean_sample - (k*standard_deviation_sample), mean_sample, standard_deviation_sample)
  proportions <- append(proportions, p)
}

# X~Exp(2)

mean <- 2
standard_deviation <- sqrt(2**2)

for(k in 1:3){
  p <- pexp(mean + (k*standard_deviation), rate = 1/2) - pexp(mean - (k*standard_deviation), rate = 1/2)
  probabilities <- append(probabilities, p)
}

sample_points <- rexp(n, rate = 1/2)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- pexp(mean_sample + (k*standard_deviation_sample), rate = 1/2) - pexp(mean_sample - (k*standard_deviation_sample), rate = 1/2)
  proportions <- append(proportions, p)
}

# X~Gamma(a = 2, b = 1)

a <- 2
b <- 1

mean <- a*b
standard_deviation <- sqrt(a*(b**2))

for(k in 1:3){
  p <- pgamma(mean + (k*standard_deviation), shape = a, scale = b) - pgamma(mean - (k*standard_deviation), shape = a, scale = b)
  probabilities <- append(probabilities, p)
}

sample_points <- rgamma(n, shape = a, scale = b)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- pgamma(mean_sample + (k*standard_deviation_sample), shape = a, scale = b) - pgamma(mean_sample - (k*standard_deviation_sample), shape = a, scale = b)
  proportions <- append(proportions, p)
}

# X~Gamma(a = 3, b = 1)

a <- 3
b <- 1

mean <- a*b
standard_deviation <- sqrt(a*(b**2))

for(k in 1:3){
  p <- pgamma(mean + (k*standard_deviation), shape = a, scale = b) - pgamma(mean - (k*standard_deviation), shape = a, scale = b)
  probabilities <- append(probabilities, p)
}

sample_points <- rgamma(n, shape = a, scale = b)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- pgamma(mean_sample + (k*standard_deviation_sample), shape = a, scale = b) - pgamma(mean_sample - (k*standard_deviation_sample), shape = a, scale = b)
  proportions <- append(proportions, p)
}

# X~Beta(a = 2, b = 2)

a <- 2
b <- 2

mean <- a/(a + b)
variance <- (a * b)/(((a + b)**2) * (a + b + 1))
standard_deviation <- sqrt(variance)

for(k in 1:3){
  p <- pbeta(mean + (k*standard_deviation), shape1 = 2, shape2 =  2) - pbeta(mean - (k*standard_deviation), shape1 =  2, shape2 = 2)
  probabilities <- append(probabilities, p)
}

sample_points <- rbeta(n, shape1 = 2, shape2 = 2)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- pbeta(mean_sample + (k*standard_deviation_sample), shape1 = 2, shape2 =  2) - pbeta(mean_sample - (k*standard_deviation_sample), shape1 = 2, shape2 =  2)
  proportions <- append(proportions, p)
}

# X~Weibull(a = 4, b = 1)

a <- 4
b <- 1 

mean <- (a)^(-1/b) * factorial(1/b) # factorial instead gamma(1/b)
variance <- ((a)^(-2/b)) * (factorial(2/b) - ((factorial(1/b))^2)) # factorial instead gamma((2/b)+1)
standard_deviation <- sqrt(variance)

for(k in 1:3){
  p <- pweibull(mean + (k*standard_deviation), shape = 4, scale = 1) - pweibull(mean - (k*standard_deviation), shape = 4, scale = 1)
  probabilities <- append(probabilities, p)
}

sample_points <- rweibull(n, shape = 4, scale = 1)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- pweibull(mean_sample + (k*standard_deviation_sample), shape = 4, scale = 1) - pweibull(mean_sample - (k*standard_deviation_sample), shape = 4, scale = 1)
  proportions <- append(proportions, p)
}

# X~Lognormal(mu = 3, v = 4)

mu <- 3
v <- 4
sigma <- sqrt(v)

mean <- exp(mu + ((sigma)^2 / 2)) 
variance <- exp(2 * mu + (sigma)^2) * (exp(sigma^2) - 1) 
standard_deviation <- sqrt(variance)

for(k in 1:3){
  p <- plnorm(mean + (k*standard_deviation), meanlog = mu, sdlog = sigma) - plnorm(mean - (k*standard_deviation), meanlog = mu, sdlog = sigma)
  probabilities <- append(probabilities, p)
}

sample_points <- rlnorm(n, meanlog = mu, sdlog = sigma)

mean_sample <- mean(sample_points)
standard_deviation_sample <- sd(sample_points)

for(k in 1:3){
  p <- plnorm(mean_sample + (k*standard_deviation_sample), meanlog = mu, sdlog = sigma) - plnorm(mean_sample - (k*standard_deviation_sample), meanlog = mu, sdlog = sigma)
  proportions <- append(proportions, p)
}



