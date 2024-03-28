# Pierre Lafaye and Benoit Liquet (2009). Understanding Convergence Concepts: A Visual-Minded and Graphical Simulation-Based Approach. DOI: 10.1198/tas.2009.0032.

  # Implementing convergence in probability approach described in Section 2.1 (Convergence in Probability), to visualize and better understand the idea of convergence in probability.

convergence_p <- function(M, nmax, epsilon){
  paths <- matrix(0, nrow = nmax, ncol = M)
  for(n in 1:nmax){
    yn <- numeric(M)
    for(m in 1:M){  # M sample paths
      sample <- rnorm(nmax, mean = 0, sd = 1)
      yn[m] <- sum(sample)/n # bar(x) = bar(y) = 1\n \sum_{i = 1}^{n} Y_{i}
    }
    paths[n,] <- yn
  }
  return(paths)
}

M <- 500
n <- 2000
epsilon <- 0.05
test <- convergence_p(M, n, epsilon)

matplot(c(1:n), test[,1:10], ylim = c(-1.8, 1.8), main = expression("Convergence of " * bar(X) * " towards " * mu),
        xlab = "n", ylab = expression(-epsilon +epsilon), type = 'l', col = "slategray")
lines(1:n, rep(epsilon,n), col = "firebrick")
lines(1:n, rep(-epsilon,n), col = "firebrick")

convergence_p <- function(M, nmax, epsilon){
  pn <- numeric(nmax)
  paths <- matrix(0, nrow = nmax, ncol = M)
  for(n in 1:nmax){
    yn <- numeric(M)
    for(m in 1:M){  # M sample paths
      sample <- rnorm(nmax, mean = 0, sd = 1)
      yn[m] <- sum(sample)/n # bar(x) = bar(y) = 1\n \sum_{i = 1}^{n} Y_{i}
    }
    paths[n,] <- yn
  }
  for(n in 1:nmax){
    count <- sum(abs(paths[n, ]) > epsilon)
    pn[n] <- count/M # frequency p = 1/M (#[xn- x] > epsilon) 
  }
  return(pn)
}

proportion <- convergence_p(M, n, epsilon)

plot(1:n, proportion, main = expression("Proportion " * hat(p)[n] * " as n grows"), 
     xlab = expression(n), ylab = expression(hat(p)[n]), type = 'l', col = "steelblue")


