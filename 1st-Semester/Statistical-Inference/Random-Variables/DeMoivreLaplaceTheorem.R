# Visualizing the De Moivre-Laplace Theorem; with p = 0.1 and A = {5, 10, 20, 50, 100, 500}.

  # De Moivre-Laplace theorem.

    # De Moivre-Laplace theorem states that, under certain conditions, a Binomial distribution approximates to a Normal distribution.
    # In particular, the theorem states that the probability mass function of a random variable X with a Binomial distribution, converges to the probability density function of the Normal distribution with mean np and standard deviation sqrt(np(1 - p)), as n grows with p [0, 1].

  # Plot the probability mass function of X~Binomial(n,p) over the density function f(x) of a Normal distribution Normal(np, npq) with n E A.

p <- 0.1
q <- 1 - p
n <- 5
xval <- c(0:n) # x values - [0, 1, 2, 3, 4, 5]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 5,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 10
xval <- c(0:n) # x values - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 10,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 20
xval <- c(0:n) # x values - [0, 1, ..., 20]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 20,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 50
xval <- c(0:n) # x values - [0, 1, ..., 50]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 50,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 100
xval <- c(0:n) # x values - [0, 1, ..., 100]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 100,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 500
xval <- c(0:n) # x values - [0, 1, ..., 500]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 500,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

  # Plot the cumulative distribution function of X~Binomial(n,p) over the cumulative distribution function F(x) of a random variable with Normal distribution Normal(np, npq) with n E A.

p <- 0.1
q <- 1 - p
n <- 5
xval <- c(0:n) # x values - [0, 1, 2, 3, 4, 5]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 5,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 10
xval <- c(0:n) # x values - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 10,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 20
xval <- c(0:n) # x values - [0, 1, ..., 20]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 20,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 50
xval <- c(0:n) # x values - [0, 1, ..., 50]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 50,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 100
xval <- c(0:n) # x values - [0, 1, ..., 100]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 100,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.1
q <- 1 - p
n <- 500
xval <- c(0:n) # x values - [0, 1, ..., 500]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 500,  p = 0.1"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

  # Repeat the experiment with p = 0.5 and p = 0.9

p <- 0.5
q <- 1 - p
n <- 50
xval <- c(0:n) # x values - [0, 1, ..., 50]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 50,  p = 0.5"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.5
q <- 1 - p
n <- 500
xval <- c(0:n) # x values - [0, 1, ..., 500]
plot(pbinom(xval, n, p),  main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 500,  p = 0.5"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.5
q <- 1 - p
n <- 50
xval <- c(0:n) # x values - [0, 1, ..., 50]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 50,  p = 0.5"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.5
q <- 1 - p
n <- 500
xval <- c(0:n) # x values - [0, 1, ..., 500]
plot(dbinom(xval, n, p),  main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 500,  p = 0.5"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.9
q <- 1 - p
n <- 50
xval <- c(0:n) # x values - [0, 1, ..., 50]
plot(dbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 50,  p = 0.9"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.9
q <- 1 - p
n <- 500
xval <- c(0:n) # x values - [0, 1, ..., 500]
plot(dbinom(xval, n, p),  main="De Moivre-Laplace Theorem", xlab = "x", ylab = "f(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 500,  p = 0.9"))), side = 3)
par(new=TRUE)
plot(dnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.9
q <- 1 - p
n <- 50
xval <- c(0:n) # x values - [0, 1, ..., 50]
plot(pbinom(xval, n, p), main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 50,  p = 0.9"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

p <- 0.9
q <- 1 - p
n <- 500
xval <- c(0:n) # x values - [0, 1, ..., 500]
plot(pbinom(xval, n, p),  main="De Moivre-Laplace Theorem", xlab = "x", ylab = "F(x)", type = "l") # parameters - x values, size, probability
mtext(substitute(paste(italic("N = 500,  p = 0.9"))), side = 3)
par(new=TRUE)
plot(pnorm(xval, (n * p), sqrt(n * p * q)), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n') # parameters - x values, mean = np, standard deviation = sqrt(npq)

  # N o t e : As the De Moivre-Laplace theorem states, a random variable with binomial distribution X~Binomial(n, p), converges to a Normal distribution X~Normal(np, sqrt(np(1 - q))), as n grows and p is between 0 and 1. 









