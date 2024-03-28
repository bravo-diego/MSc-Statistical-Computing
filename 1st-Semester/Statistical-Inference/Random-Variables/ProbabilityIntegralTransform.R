# Probability Integral Transform

y <- runif(100, 0, 1)
x <- -2 * log(1 - y)

hist(x, breaks = 10,  col = "steelblue", main = "Histogram of x Values", xlab = "x values", ylab = "Frecuency")

  # As the theorem states the distribution of the transformed values are consistent with what would be expected from the transformation based on the formula x = −2log(1 − y); X~Exp(lambda).
