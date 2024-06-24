# Principal Factor Method Estimation

# Let S be the variance-covariance matrix defined as 

x1 <- c(0.35, 0.15, -0.19)
x2 <- c(0.15, 0.13, -0.03)
x3 <- c(-0.19, -0.03, 0.16)

S <- cbind(x1, x2, x3) # S matrix

# Assuming a 2-factor model, estimate the model parameters using the principal factor method.

ev <- eigen(S)
values <- ev$values # eigenvalues
vectors <- ev$vectors # eigenvectors

L <- c() # empty matrix

for (i in 1:3) {
  L <- c(L, t(matrix(sqrt(values[i])%*%(vectors[, i]))))
}

L <- matrix(L, nrow = 3) # estimated loading matrix

Psi <- diag(S - L%*%t(L)) # estimated Psi matrix

communalities <- diag(L[1:3,]%*%t(L[1:3,])) # communalities
