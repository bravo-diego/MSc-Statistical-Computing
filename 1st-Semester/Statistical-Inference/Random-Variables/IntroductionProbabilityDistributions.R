# Introduction to Probability Distributions in R

  # Discrete Uniform Distribution

    # Probability Mass Function 

x <- 1:10 # grid of values
fx <- dunif(x, min=0, max=10) # uniform density function; (x-axis values, lower limit, upper limit)
fx
par(mgp=c(1.9,0.6,0),mar=c(3,3,2,1)+0.1) # modifying space between axis/ticks labels
plot(fx, main = "PMF Discrete Uniform", xlab = "x", ylab = "f(x)", ylim = c(0, 0.5))

    # Cumulative Distribution Function

x <- 0:11 # grid of values
n <- length(x)
fx <- punif(x, min=0, max=10) # continuous uniform distribution (x-axis values, lower limit, upper limit)
fx
plot(x = NA, y = NA, pch = NA, xlim = c(-0.2, 11.0), ylim = c(-0.01, 1), main = "CDF Discrete Uniform", xlab = "x", ylab = "F(x)") 
for(i in 1:(n)) points(x=x[i+0:1], y=fx[c(i,i)], type="l")

  # sample Function

help("sample")

  # Sample takes a sample of the specified size from the elements of x using either with or without replacement

# sample(x, size, replace = FALSE, prob = NULL) 

  # Parameters

    # x -- either a vector of one or more elements from which to choose, or a positive integer.

    # n -- a positive number, number of items to choose from.

    # size -- a non negative integer giving the number of items to choose.

    # replace -- should sample be with replacement?

    # prob -- a vector of probability weights for obtaining the elements of the vector being sampled.

  # Using the sample function simulate a sample of size 10,000 from the distribution U(1,...,10). Fix the seed to 13, show results in a frequency table, and calculate the mean and variance.

set.seed(13)
U <- 1:10 # distribution of X
results_simulation <- sample(U, 10000, replace=TRUE) # parameters - vector of one or more elements from which to choose; sample size 10; replacement TRUE (i.e. probabilities won't change between picks)
results_table <- table(results_simulation) # show up results in a table
mean <- mean(results_simulation) # mean 
variance <- var(results_simulation) # variance 
standard_deviation <- sd(results_simulation) # standard deviation

# Results
print(results_table) 
print(mean)
cat(variance, standard_deviation)

plot(results_table , main = "Sample U Frequencies", xlab = "U", ylab = "Frequency", ylim = c(0, 2000), type = 'p', xaxt = "n", yaxt = "n")
# specifying the range and labels of tick marks
axis(1, at = c(2, 4, 6, 8, 10))
axis(2, at = c(0, 500, 1000, 1500, 2000))

  # Using the sample function simulate 10 flips of a coin and count the number of tails. Repeat this process 10^4 times and show the first results. Plot the frequencies and proportions of the number of tails obtained in all the experiments. 

V <- c('H', 'T') # possible values of x
Results <- sample(V, 10, replace = TRUE) # parameters - vector of one or more elements from which to choose; sample size 10; replacement TRUE (i.e. probabilities won't change between picks)
print(Results) # number of heads and tails obtained
print(table(Results)) # summary of the experiment (# Heads, # Tails)
print(table(Results)[1]) # number of heads obtained

a <- 0
fqns <- c() # empty vector to store frequencies
for (i in 1:10000) { # loop for repeat the experiment 10^4 times
  Results <- sample(V, 10, replace = TRUE)
  Heads <- table(Results)
  if (a < 3) { # print out the first 3 results
    print(Heads)  # number of Heads and Tails obtained
    print(Heads[1]) # number of Heads obtained
  }
  a <- a + 1
  if (Heads[1] == 10 && Results[1] == 'T'){
    Heads[1] <- 0 
  }
  fqns <- append(fqns, Heads[1]) # update frequencies vector with the number of Heads obtained
}

Frequencies <- table(fqns) # save frequencies in a table
df <- as.data.frame(Frequencies) # convert table to data frame
print(df) 

# Bar plot of Heads Frequencies
barplot(df$Freq, main = "Heads Frequencies", xlab = "Number of Heads", ylab = "Frecuency", ylim = c(0, 3000), names.arg = df$fqns, col="lightblue", border = NA)
box()

# Bar plot of Head proportions
N <- sum(df$Freq) # N = 10,000
df$Freq <- as.numeric(as.character(df$Freq)) / N
p <- sum(df$Freq) # p = 1
barplot(df$Freq, main = "Heads Proportions", xlab = "Number of Heads", ylab = "Proportion", ylim = c(0, 0.25), names.arg = df$fqns, col="lightblue", border = NA)
box()

  # Using the dbinom function plot the probability mass function of a variable with distribution B(10,0.5) over the proportions plot did before.

# Bar plot of Head proportions and binomial probability mass function
Values <- c(0:10)
barplot(df$Freq, main = "Heads Proportions", xlab = "Number of Heads", ylab = "Proportion", ylim = c(0, 0.26), names.arg = df$fqns, col="lightblue", border = NA)
par(new=TRUE)
plot(0:10, dbinom(Values, size = 10, prob = 0.5), xlab = NA, ylab = NA, xaxt = 'n', yaxt = 'n', type = 'p')
legend('topright', legend = c('Proportions', 'Binomial Distribution'), col = c('lightblue', 'black'), lwd = 4, cex = 0.4, horiz = TRUE, bty = "n")

  # Repeat the process with a coin having a probability of 0.3 of getting Heads.

V <- c('H', 'H', 'H', 'T', 'T', 'T', 'T', 'T', 'T', 'T') # probability of getting Heads 0.3
Results <- sample(V, 10, replace = TRUE)  # parameters - vector of one or more elements from which to choose; sample size 10; replacement TRUE (i.e. probabilities won't change between picks)
print(Results) # number of heads and tails obtained
print(table(Results)) # summary of the experiment (# Heads, # Tails)
print(table(Results)[1]) # number of heads obtained

a <- 0
fqns <- c() # empty vector to store frequencies
for (i in 1:10000) { # loop for repeat the experiment 10^4 times
  Results <- sample(V, 10, replace = TRUE)
  Heads <- table(Results)
  if (a < 3) { # print out the first 3 results
    print(Heads) # number of Heads and Tails obtained
    print(Heads[1]) # number of Heads obtained
  }
  a <- a + 1
  if (Heads[1] == 10 && Results[1] == 'T'){
    Heads[1] <- 0 
  }
  fqns <- append(fqns, Heads[1]) # update frequencies vector with the number of Heads obtained
}

Frequencies <- table(fqns) # save frequencies in a table
df <- as.data.frame(Frequencies) # convert table to data frame
df$fqns <- as.character(df$fqns)
row <- c(10, 0)
df <- rbind(row, df)
df <- df[c(2:11,1:1),]
print(df) 

# Bar plot of Heads Frequencies
barplot(df$Freq, main = "Heads Frequencies", xlab = "Number of Heads", ylab = "Frecuency", ylim = c(0, 2800), names.arg = df$fqns, col = "lightpink", border = NA)
box()

N <- sum(df$Freq) # N = 10,000
df$Freq <- as.numeric(as.character(df$Freq)) / N
p <- sum(df$Freq) # p = 1

# Bar plot of Head proportions
barplot(df$Freq, main = "Heads Proportions", xlab = "Number of Heads", ylab = "Proportion", ylim = c(0, 0.3), names.arg = df$fqns, col = "lightpink", border = NA)
box()

# Bar plot of Head proportions and binomial probability mass function
Values <- c(0:10)
barplot(df$Freq, main = "Heads Proportions", xlab = "Number of Heads", ylab = "Proportion", ylim = c(0, 0.3), names.arg = df$fqns, col = "lightpink", border = NA)
par(new=TRUE)
plot(0:10, dbinom(Values, size = 10, prob = 0.3), xlab = NA, ylab = NA, xaxt = 'n', yaxt = 'n', type = 'p')
legend('topright', legend = c('Proportions', 'Binomial Distribution'), col = c('lightpink', 'black'), lwd = 5, cex = 0.5, horiz = TRUE, bty = "n")

  # An urn contains 46 gray marbles and 49 white marbles. Using the sample function, simulate the drawn without replacement of 20 of these marbles and count the number of gray marbles obtained. Repeat the experiment 10,000 times and plot the frequencies of gray marbles obtained in every experiment. What's the probability of obtain 5 grey marbles when draw 20 marbles? Plot the proportion of grey marbles obtained in the experiments and over this plot show the Hypergeometric Distribution.

U <- c('Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr',
       'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr',
       'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr',
       'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr',
       'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Gr', 'Wh', 'Wh', 'Wh', 'Wh', 
       'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh',
       'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh',
       'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh',
       'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh', 'Wh',
       'Wh', 'Wh', 'Wh', 'Wh', 'Wh') # 46 grey marbles; 49 white marbles

Results <- sample(U, 20, replace = FALSE)  # parameters - vector of one or more elements from which to choose; sample size 20; replacement FALSE (i.e. probabilities will change between picks)
print(Results) # number of heads and tails obtained

fqns <- c() # empty vector to store frequencies
for (i in 1:10000) { # loop for repeat the experiment 10^4 times
  Results <- sample(U, 20, replace = FALSE)
  GMarbles <- table(Results)
  if (GMarbles[1] == 20 && Results[1] == 'Wh'){
    GMarbles[1] <- 0 
  }
  fqns <- append(fqns, GMarbles[1]) # update frequencies vector with the number of grey marbles obtained
}

Frequencies <- table(fqns) # save frequencies in a table
df <- as.data.frame(Frequencies) # convert table to data frame
print(df) 

# Bar plot of Grey Marbles Frequencies
barplot(df$Freq, main = "Grey Marbles Frequencies", xlab = "Number of Grey Marbles", ylab = "Frecuency", ylim = c(0, 2100), names.arg = df$fqns)
box()

  # dhyper Function

help("Hypergeometric")  

  # Density function for the hypergeometric distribution.

# dhyper(x, m, n, k, log = FALSE)

  # Parameters

    # x -- vector of quantiles representing the number of grey balls drawn without replacement from an urn which contains both grey and white balls.

    # m -- the number of grey balls in the urn.

    # n -- the number of white balls in the urn.

    # k -- the number of balls drawn from the urn.

# What's the probability of getting 5 grey marbles in 20 draws?
print((dhyper(5, 46, 49, 20))) # parameters - vector representing the number of grey marbles drawn without replacement; number of grey marbles; number of white marbles; number of marbles drawn from the urn

# Bar plot of Grey Marbles proportions and hypergeometric probability mass function
print((dhyper(c(0:20), 46, 49, 20)))
N <- sum(df$Freq) # N = 10,000
df$Freq <- as.numeric(as.character(df$Freq)) / N
p <- sum(df$Freq) # p = 1
barplot(df$Freq, main = "Grey Marbles Proportions", xlab = "Number of Grey Marbles", ylab = "Proportion", ylim = c(0, 0.21), names.arg = df$fqns)
par(new=TRUE)
plot((dhyper(c(0:20), 46, 49, 20)), xlab = NA, ylab = NA, xaxt = 'n', yaxt = 'n', type = 'p')
legend('topright', legend = c('Proportions', 'Hypergeometric Distribution'), col = c('darkgray', 'black'), lwd = 5, cex = 0.4, horiz = TRUE, bty = "n")

  