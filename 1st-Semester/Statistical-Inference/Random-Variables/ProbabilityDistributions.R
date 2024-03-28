# Probability Distributions in R


# Geometric Distribution

  # Using the sample function, write a function that takes the number of throws of N and a probability of getting Heads p; the function have to return a vector containing all the outcomes before getting Heads (vector of length N).   

coin <- function(p, N){ # parameters - probability p; number of throws N
  x <- c('H', 'T') # possible outcomes
  v <- c() # empty vector for store frequencies
  for (i in 1:N){ # loop for repeat the experiment N times
    c <- 1
    result <- sample(x, 1, replace=TRUE, c(p, (1 - p))) # parameters - vector of one or more elements from which to choose; sample size 1; replacement TRUE (i.e. probabilities won't change between picks); probability p of get Heads, probability 1 - p of get Tails
    while(result[1] != 'H'){ # while loop for interrupt the coin flips
      result <- sample(x, 1, replace=TRUE, c(p, (1 - p)))
      c = c + 1 # number of throws taken until get Heads
    }
    v <- append(v, c)
  }
  return(v)
} 

  # Using the function written before simulate a random variable with Geometric distribution X~Geom(p) with p = 0.5, 0.1, 0.01 and N = 10^4. Plot the proportions and over this plot add the probability mass function associated with this variable.

# random variable ~Geom(0.5)
result <- coin(0.5, 10000)
max_number_flips <- max(result)
Geom_0.5 <- table(result) 

plot(prop.table(Geom_0.5), main="Number of flips until the first Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("N = 10"^4))), side = 3) # subtitle
par(new=TRUE)
plot(dgeom(c(1:max_number_flips), 0.5), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~Geom(0.5)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")

# random variable ~ Geom(0.1)
result <- coin(0.1, 10000)
max_number_flips <- max(result)
Geom_0.1 <- table(result) 

plot(prop.table(Geom_0.1), main="Number of flips until the first Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("N = 10"^4))), side = 3) # subtitle
par(new=TRUE)
plot(dgeom(c(1:max_number_flips), 0.1), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~Geom(0.1)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")

# random variable ~ Geom(0.01) 
result <- coin(0.01, 10000)
max_number_flips <- max(result)
Geom_0.01 <- table(result) 

plot(prop.table(Geom_0.01), main="Number of flips until the first Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("N = 10"^4))), side = 3) # subtitle
par(new=TRUE)
plot(dgeom(c(1:length(Geom_0.01)), 0.01), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~Geom(0.01)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")

  # Repeat the experiment with N = 10^6. Calculate the mean and standard deviation. 

# random variable ~ Geom(0.5) 
result <- coin(0.5, 1000000)
max_number_flips <- max(result) 
Geom_0.5 <- table(result) 
mean(result) # mean
sd(result) # standard deviation

plot(prop.table(Geom_0.5), main="Number of flips until the first Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("N = 10"^6))), side = 3) # subtitle
par(new=TRUE)
plot(dgeom(c(1:max_number_flips), 0.5), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~Geom(0.5)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")

# random variable ~ Geom(0.1) 
result <- coin(0.1, 1000000)
max_number_flips <- max(result) 
Geom_0.1 <- table(result) 
mean(result) # mean
sd(result) # standard deviation

plot(prop.table(Geom_0.1), main="Number of flips until the first Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("N = 10"^6))), side = 3) # subtitle
par(new=TRUE)
plot(dgeom(c(1:max_number_flips), 0.5), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~Geom(0.1)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")

# random variable ~ Geom(0.01) 
result <- coin(0.01, 1000000)
max_number_flips <- max(result) 
Geom_0.01 <- table(result) 
mean(result) # mean
sd(result) # standard deviation

plot(prop.table(Geom_0.01), main="Number of flips until the first Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("N = 10"^6))), side = 3) # subtitle
par(new=TRUE)
plot(dgeom(c(1:max_number_flips), 0.01), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~Geom(0.01)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")


# Negative Binomial Distribution

  # Write a function that takes as parameters the number of throws (N) of a coin until get r heads, and a probability of getting Heads denoted by p. The function has to return a vector of length N with the number of throws needed to get r Heads in N experiments. Plot the proportions with N = 10^6 and p = 0.2, 0.1 and r = 2, 7 and over this plot add the probability mass function associated with this variable.

rcoins <- function(p, N, r){ # function parameters - probability p; number of throws N; number of successes wanted r
  x <- c('H', 'T') # possible outcomes
  v <- c() # empty vector for store frequencies
  for (i in 1:N){ # loop for repeat the experiment N times
    c <- 0
    h <- r # number of Heads
    while (h != 0){
      result <- sample(x, 1, replace=TRUE, c(p, (1 - p))) # parameters - vector of one or more elements from which to choose; sample size 1; replacement TRUE (i.e. probabilities won't change between picks); probability p of get Heads, probability 1 - p of get Tails
      if (result[1] == 'H'){
        h = h - 1 # number of Heads remaining
        c = c + 1 # number of throws taken until get r Heads
      }
      else{
        c = c + 1
      }
    }
    v <- append(v, c)
  }
  return(v)
}

# random variable ~BN(2, 0.2)
result <- rcoins(0.2, 1000000, 2)
max_number_flips <- max(result)
BN_0.2 <- table(result)

plot(prop.table(BN_0.2), main="Number of flips until r Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("r = 2,  N = 10"^6))), side = 3) # subtitle
par(new=TRUE)
plot(dnbinom(c(2:(max_number_flips-2)), 2, 0.2), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~BN(2, 0.2)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")

# random variable ~BN(7, 0.1)
result <- rcoins(0.1, 1000000, 7)
max_number_flips <- max(result)
BN_0.1 <- table(result)

plot(prop.table(BN_0.1), main="Number of flips until r Heads", xlab = "Number of flips", ylab = "Proportion", col="lightblue")
mtext(substitute(paste(italic("r = 7,  N = 10"^6))), side = 3) # subtitle
par(new=TRUE)
plot(dnbinom(c(7:max_number_flips-7), 7, 0.1), xlab=NA, ylab=NA, xaxt = 'n', yaxt = 'n', col="firebrick2")
legend('topright', legend = c('Proportions', '~BN(7, 0.1)'), col = c('lightblue', 'firebrick2'), lwd = 5, cex = 0.75, horiz = TRUE, bty = "n")





