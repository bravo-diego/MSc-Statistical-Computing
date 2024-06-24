# Factor Analysis

# Factor Analysis (FA) can be considered an extension of Principal Component Analysis (PCA). Both methods reduce the dimension of a set of variables into a smaller number of factors which captures most of the variance of the original dataset.

# FA tries to explain the correlation between variables by considering both a common source of variance shared among the variables and a specific source of variance unique to each variable.

  # The main difference between PCA and FA is that PCA is primarily a descriptive tool, whereas FA is a formal statistical model.

# Let R be a 4 by 4 matrix which shows the sample correlations between 4 variables related to the financial status of a company:

x1 <- c(1, 0.69, 0.28, 0.35)
x2 <- c(0.69, 1, 0.255, 0.195)
x3 <- c(0.28, 0.255, 1, 0.61)
x4 <- c(0.35, 0.195, 0.61, 1)

R <- cbind(x1, x2, x3, x4) # R matrix

# Calculate the eigenvalues and eigenvectors of R.

ev <- eigen(R)
values <- ev$values # eigenvalues
vectors <- ev$vectors # eigenvectors

# Formulate the orthogonal factor model with m factors for the vector X that generated these data.

L <- c() # empty matrix

for (i in 1:4) {
  L <- c(L, t(matrix(sqrt(values[i])%*%(vectors[, i]))))
}

L <- matrix(L, nrow = 4) # loading matrix
Psi <- diag(R - L%*%t(L)) # Psi matrix

R_estimated <-  L%*%t(L) + Psi # covariance matrix S = LL' + Psi

# For m = 2 and m = 3 calculate the loading matrices, the communalities, and the percentage that the communality represents of the variance of each variable.

var_proportion <- values / 4
x <- c(1, 2, 3, 4)
variance <- c(0, 0, 0, 0)
t <- 0

for (i in 1:4) {
  t <- t + var_proportion[i]
  variance[i] <- t
}

plot(x, variance, type = "l", main = "Scree Plot",
     xlab = "Factor Number", ylab = "Eigenvalue", bty = "L") # score plot; first two factors explain approx 80% of the total variance

  # Communalities

comm_m2 <- diag(L[1:4, 1:2]%*%t(L[1:4, 1:2])) # m = 2
comm_m3 <- diag(L[1:4, 1:3]%*%t(L[1:4, 1:3])) # m = 3




