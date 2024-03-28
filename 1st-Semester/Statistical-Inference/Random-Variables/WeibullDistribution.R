# Weibull Distribution  
  
  # Let X ~ Weibull(alpha, beta) with alpha values a = {1, 2, 4} and beta values b = {2, 2, 3}; plot the X distribution.

alpha <- 1 # alpha value
beta <- 2 # beta value

n <- 1000
sample <- rweibull(n, shape = alpha, scale = beta)
hist(sample, breaks = 20, main = 'Histogram of x Values', xlab = 'Sample values', ylab = 'Frequency', col = 'steelblue')
mtext(substitute(paste(italic("X ~ Weibull(1, 2)"))), side = 3)

alpha <- 2
beta <- 2

n <- 1000
sample <- rweibull(n, shape = alpha, scale = beta)
hist(sample, breaks = 20, main = 'Histogram of x Values', xlab = 'Sample values', ylab = 'Frequency', col = 'steelblue')
mtext(substitute(paste(italic("X ~ Weibull(2, 2)"))), side = 3)

alpha <- 4
beta <- 3

n <- 1000
sample <- rweibull(n, shape = alpha, scale = beta)
hist(sample, breaks = 20, main = 'Histogram of x Values', xlab = 'Sample values', ylab = 'Frequency', col = 'steelblue')
mtext(substitute(paste(italic("X ~ Weibull(3, 4)"))), side = 3)

  # Plot the Hazard and Survivor Functions X~Weibull(alpha, beta). N o t e : Consider the alpha and beta values listed above.

alpha <- 1
beta <- 2

x <- seq(0, 10, by = 0.1)
Rt <- 1 - pweibull(x, shape = alpha, scale = beta)
ht <- dweibull(x, shape = alpha, scale = beta) / Rt

plot(x, Rt, type = 'l', main = 'Survivor Function', xlab = 'x', ylab = 'R(t)')
mtext(substitute(paste(italic("alpha = 1, beta = 2)"))), side = 3)

plot(x, ht, type = 'l', main = 'Hazard Function', xlab = 'x', ylab = 'h(t)')
mtext(substitute(paste(italic("alpha = 1, beta = 2)"))), side = 3)

alpha <- 2
beta <- 2

x <- seq(0, 10, by = 0.1)
Rt <- 1 - pweibull(x, shape = alpha, scale = beta)
ht <- dweibull(x, shape = alpha, scale = beta) / Rt

plot(x, Rt, type = 'l', main = 'Survivor Function', xlab = 'x', ylab = 'R(t)')
mtext(substitute(paste(italic("alpha = 2, beta = 2)"))), side = 3)

plot(x, ht, type = 'l', main = 'Hazard Function', xlab = 'x', ylab = 'h(t)')
mtext(substitute(paste(italic("alpha = 2, beta = 2)"))), side = 3)

alpha <- 3
beta <- 4

x <- seq(0, 10, by = 0.1)
Rt <- 1 - pweibull(x, shape = alpha, scale = beta)
ht <- dweibull(x, shape = alpha, scale = beta) / Rt

plot(x, Rt, type = 'l', main = 'Survivor Function', xlab = 'x', ylab = 'R(t)')
mtext(substitute(paste(italic("alpha = 3, beta = 4)"))), side = 3)

plot(x, ht, type = 'l', main = 'Hazard Function', xlab = 'x', ylab = 'h(t)')
mtext(substitute(paste(italic("alpha = 3, beta = 4)"))), side = 3)

