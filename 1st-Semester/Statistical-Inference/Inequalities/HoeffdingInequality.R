# Let alpha = 0.05 and p = 0.4. Conduct a simulation study to see how often the interval contains p (called the coverage). Do this for n = (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000); a) plot the coverage versus n, and b) plot the length of the interval versus n.  

  # N o t e s:

    # Let X1, X2, ..., Xn ~ Bernoulli(p) iid, then for all epsilon > 0 
  
      # P(|Xn - p| > epsilon) <= 2e^{-2nepsilon^2} - Hoeffding inequality

    # Consider coverage = (pn - epsilon, pn + epsilon) with epsilon = sqrt((1/(2*n))*log(2/alpha)).

# Plot the coverage versus n. 

alpha <- 0.05
p <- 0.4

N <- c(10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
S <- 1000 # number of simulations S

coverage <- numeric(length(N)) #empty vector
len <- numeric(length(N)) # empty vector to store lengths

for (i in 1:length(N)){
  n <- N[i]
  c <- numeric(S) # empty vector
  
  for (j in 1:S){
    X <- rbinom(n, 1, p)
    epsilon <- sqrt((1/(2*n))*log(2/alpha)) # epsilon value
    pn <- mean(X) # p^ value
    
    lower_bound <- pn - epsilon
    upper_bound <- pn + epsilon
    
    c[j] <- as.numeric(lower_bound <= p & p <= upper_bound)
  }
  coverage[i] <- mean(c) # proportion of values that are inside of the lower and upper bounds
  len[i] <- 2*epsilon
}

plot(N, coverage, type = 'p', main = "Coverage vs n", xlab = "n", ylab = "Coverage", pch = 18)
mtext(substitute(paste(italic("How often the interval contains p?"))), side = 3)

# Plot the length of the interval versus n. 

plot(N, len, type='l', main = "Interval length vs n", xlab = "n", ylab = "Length", col = 'steelblue3')
lines(c(min(N), max(N)), c(0.05, 0.05), col = "firebrick2", lty = 2)
legend('topright', legend = c('Interval length', 'Threshold below 0.05'), col = c('steelblue3', 'firebrick2'), lwd = 4, cex = 0.70, horiz = TRUE, bty = "n")
