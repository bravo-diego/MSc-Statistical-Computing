# Manually simulating Poisson Process in R

  # Basic Concepts of the Poisson Process - write a function that simulates a poisson process with the following assumptions:

     # Independent random variables and non-overlapping intervals

     # Random variables has the same probability distribution; distribution only is affected by the length of the interval
  
     # if Δt -> 0 then pi(Δt) ~ λΔt  Δt > 0

     # ∑pk(Δt) ~ 0 from k=2 to infinity

     # X0 = 0; p0(0) = 1

# Poisson Process Function
poisson_process <- function(lambda, t, dt){ # parameters 
  r <- (t/dt) # e.g. T/dt = 1000
  v <- numeric(r + 1) # numeric vector
  v[1] = 0 # assumption No.1 - X0 = 0; p0(0) = 1
  
  x <- c(1, 0)
  c <- replicate(r, rbinom(x, 1, (lambda*(dt + 1e-6)))) # set of random variables ~B(p)
  
  for(i in 1:r){
    v[i + 1] <- sum(c[1:i]) # cumulative sums of Bernoulli variables
  }
  
  return(v)
  
}

poisson_process(2, 10, 0.01)

r <- 10000
interval <- seq(0, 10, by = 0.001)

# Simulations
fst_simulation <-poisson_process(2, 10, 0.001) # store poisson_process results
snd_simulation <-poisson_process(2, 10, 0.001) 
trd_simulation <-poisson_process(2, 10, 0.001) 


plot(interval, fst_simulation, main = 'Poisson Proccess',
     xlab = 't', ylab = 'Event Occurrences', type = 'l', col = "dodgerblue")
mtext(substitute(paste(italic("lambda = 2"))), side = 3)
par(new=TRUE)
plot(interval, snd_simulation, main = 'Poisson Proccess',
     xlab = 't', ylab = 'Event Occurrences', type = 'l', col = "firebrick1", xaxt = 'na', yaxt = 'na')
mtext(substitute(paste(italic("lambda = 2"))), side = 3)
par(new=TRUE)
plot(interval, trd_simulation, main = 'Poisson Proccess',
     xlab = 't', ylab = 'Event Occurrences', type = 'l', col = "darkorange", xaxt = 'na',  yaxt = 'na')
mtext(substitute(paste(italic("lambda = 2"))), side = 3)

freq <- c() # empty vector 
for (i in 1:10000){ # loop for run the poisson process 10,000 times
  process <- poisson_process(0.5, 1, 0.0001)
  f <- max(process)
  freq <- append(f, freq) # storing frequencies in vector freq
}

frequencies <- table(freq)

# Poisson distribution parameters
lambda <- 0.5
l <- length(frequencies)
x <- 0:l

barplot(prop.table(frequencies), main = 'Poisson Distribution', xlab = 'Occurrences', ylab = 'Proportion', col = "dodgerblue", space = 0) # plotting frequencies
mtext(substitute(paste(italic("lambda = 0.5"))), side = 3)
par(new=TRUE)
plot(dpois(x, lambda), xlab = "", ylab = "", xaxt = 'na', yaxt = 'na', col = 'firebrick1', type = 'l')

