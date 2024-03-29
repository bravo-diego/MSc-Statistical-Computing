# Given a Bivariate Normal population with mu_{1} = 0, mu_{2} = 2, sigma_{11} = 2, sigma_{22} = 1, and rho_{12} = 0.5 determine and plot the probability ellipse that contains 50% of the probability.

library(MASS)

rho <- 0.5
sigma <- matrix(c(2, 1/sqrt(2), 1/sqrt(2), 1), 2)  # variance-covariance matrix

eigenv <- eigen(sigma) # eigenvalues and eigenvectors
print(eigenv) # eigenvectors determine the direction of the axes and the eigenvalues their length

# Simulate from a Multivariate Normal Distribution

help("mvrnorm")

n <- 10000 # no. of samples 
mu <- c(0, 2) # vector giving the means of the variables
sigma <- sigma # variance-covariance matrix

data <- mvrnorm(n, mu, sigma) 
distances <- mahalanobis(data, colMeans(data), cov(data)) # measure between a sample point and a distribution; D^{2} = (x - \mu){'}\Sigma^{-1}(x - \mu)

plot(data, pch = ".", xlab = "x1", ylab = "x2")
mytitle = "Probability Ellipse"
mysubtitle = "Expected to Contain 50% of the Observed Data"
mtext(side=3, line=2, at=0.8, adj=0.95, cex=1.2, mytitle)
mtext(side=3, line=1, at=0.3, adj=0.60, cex=1, mysubtitle)
points(data[distances > 1.386,], pch = 20, cex = 0.9, col = 'Steel Blue 2')
points(data[(distances - 0.6) < distances & distances < (1.386+0.6),], pch = 20, cex = 0.9, col = 'Orange 2')





