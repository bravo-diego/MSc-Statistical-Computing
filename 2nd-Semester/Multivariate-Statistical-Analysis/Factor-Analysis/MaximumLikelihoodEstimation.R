# Maximum Likelihood Estimation of Factor Analysis.

library(MASS)
library(psych)

# Let the loading matrix be 

l1 <- c(0.9, 0.8, 0.2, 0.3, 0.7)
l2 <- c(0.05, 0.3, 0.95, 0.9, 0.15)

L <- cbind(l1, l2) # loading matrix

  # and the specific variance matrix 

Psi <- matrix(c(
  .2, 0,0,0,0,
  0,.3,0,0,0,
  0,0,.1,0,0,
  0,0,0,.2,0,
  0,0,0,0,.3), nrow =5 , byrow = TRUE) # Psi matrix

# With μ = 0, generate a sample of size n = 1000 from X and obtain the maximum likelihood estimation of the loading matrix for m = 2 factors. Calculate the corresponding estimation of the specific variances matrix.

  # Note: X follows a normal distribution.

S <- L%*%t(L) + Psi

n <- 1000 # sample size
mu <- c(0, 0, 0, 0, 0) # mean vector

help("mvrnorm") # simulate from a multivariate normal distribution; produces one or more samples from the specified multivariate normal distribution

random_sample <- MASS::mvrnorm(n, mu, S) # random sample multivariate normal distribution
head(random_sample) # first 5 rows 

help("factanal") # perform maximum-likelihood factor analysis on a covariance matrix or data matrix

m = 2 # factors
estimation <- factanal(random_sample, m, rotation = "varimax")
print(estimation$loadings) # loadings
print(estimation$uniquenesses) # specific variance

l1_estimated <- c(0.880, 0.791, 0.158, 0.292, 0.781)
l2_estimated <- c(0.114, 0.298, 0.985, 0.827, 0.196)

L_estimated <- cbind(l1_estimated, l2_estimated) # estimated loading matrix

Psi_estimated <- Psi <- matrix(c(
  .2125879, 0,0,0,0,
  0,.2853924,0,0,0,
  0,0,.0050000,0,0,
  0,0,0,.2303148,0,
  0,0,0,0,.3515354), nrow =5 , byrow = TRUE) # estimated Psi matrix

S <- L_estimated%*%t(L_estimated) + Psi_estimated
