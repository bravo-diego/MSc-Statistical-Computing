# Given the following data: plot the pairwise scatter plot; calculate \bar{x_{i}} and s_{i} for i = [1, 2]; calculate the centering matrix P, the variance-covariance matrix S and the correlation matrix R.

library(foreign)
library(ggplot2)
library(psych)
library(scatterplot3d)
library(GGally)

data <- matrix(c(
  1, 8.7, 0.3, 3.1,
  2, 14.3, 0.9, 7.4,
  3, 18.9, 1.8, 9.0,
  4, 19.0, 0.8, 9.4,
  5, 20.5, 0.9, 8.3,
  6, 14.7, 1.1, 7.6,
  7, 18.8, 2.5, 12.6,
  8, 37.3, 2.7, 18.1,
  9, 12.6, 1.3, 5.9,
  10, 25.7, 3.4, 15.9
), ncol = 4, byrow = TRUE)

colnames(data) <- c("id", "X1", "X2", "X3")

print(data)

X <- data[, c("X1", "X2", "X3")]

ggpairs(X) # the pair plot help us to visualize the distribution of single variables as well as relationships between two variable;

means <- apply(X[, -1], 2, mean)
print(means) # measure of central tendency; average of a given set of values

var <- apply(X[, -1], 2, var)
print(var) # statistical measurement of the spread between numbers in a data set

print(cov(X[, "X1"], X[, "X2"])) # measure of the joint variability of two random variables
print(cor(X[, "X1"], X[, "X2"])) # statistical measure of the strength of a linear relationship between two variables

n <- nrow(X) # no. of rows
P <- diag(n) - (1/n)*matrix(1, n, n) 
print(P) # P matrix; centering matrix plays an important role since we will use it to remove the column means from a matrix, centering the matrix

x_bar <- colMeans(X)
print(x_bar) # the mean vector consists of the means of each variable; x_bar dimension 1 row by 3 columns

S <- (1/n)*t(X)%*%P%*%X
print(S) # the variance-covariance matrix consists of the variances of the variables along the main diagonal and the covariances between each pair of variables in the other matrix positions

D <- diag(diag(S))

D_inv <- diag(1 / sqrt(diag(D)))

R <- D_inv %*% S %*% D_inv
print(R) # correlation coefficients between variables


