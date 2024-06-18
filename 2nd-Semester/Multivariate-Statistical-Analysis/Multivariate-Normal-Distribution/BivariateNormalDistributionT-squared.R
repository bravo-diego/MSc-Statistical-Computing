# Verify that T^{2} remains unchanged if each observation x_{j}, j = 1, 2, 3; is replaced by Cx_{j}, with C = (1 -1\\ 1 1), given a random sample of a Bivariate Normal Distribution.

X <- matrix(c(6, 10, 8, 9, 6, 3), 3) # random sample of a bivariate normal distribution with size n = 3

mu <- colMeans(X) # mean vector
print(mu)

n <- nrow(X) # no. of rows
P <- diag(n) - (1/n)*matrix(1, n, n) # P matrix; centering matrix plays an important role since we will use it to remove the column means from a matrix, centering the matrix

S <- t(X)%*%P%*%X/(n-1) # the variance-covariance matrix consists of the variances of the variables along the main diagonal and the covariances between each pair of variables in the other matrix positions
print(S) 
  
C <- matrix(c(1, 1, -1, 1), 2) # C matrix
X_transformed <- t(C%*%t(X)) # Cx_{j} transformation

mu_transformed <- colMeans(X_transformed) # mean vector
print(mu_transformed)

S_transformed <- t(X_transformed)%*%P%*%X_transformed/(n-1) # the variance-covariance matrix consists of the variances of the variables along the main diagonal and the covariances between each pair of variables in the other matrix positions
print(S_transformed)

  # By solving T^{2} = n(\bar{x} - \mu)^{'}S^{-1}(\bar{x} - \mu) we found that T^{2} remains the same despite the mean vector and S matrix transformation.
