# Consider the following experiment: a coin (A or B) is randomly chosen, and then a coin toss is performed with that coin, with the possible outcomes being Heads (Sol) and Tails (Águila). The result of this experiment is:

  # 1) A Heads, 2) B Tails, 3) A Tails, 4) B Heads, 5) B Tails, 6) NA, Tails

# If we parametrize as follows:
  
  # theta: Probability that the coin is A.
  # theta_A: Probability that the result is Heads given that the coin is A.
  # theta_B: Probability that the result is Heads given that the coin is B.

# Use the EM algorithm techniques to estimate the parameters theta, theta_A, and theta_B, considering the presence of missing data.

# In this case, the selected coin in 6) is a latent variable, as it cannot be directly observed or measured, but it influences the observable outcomes of the experiment.

theta <- 0.5
theta_A <- 0.5
theta_B <- 0.5 # initialize parameters

theta_A_given_Aguila <- function(theta, theta_A, theta_B) {
  theta * (1 - theta_A) / (theta * (1 - theta_A) + (1 - theta) * (1 - theta_B))
} # Bayes theorem - (probability A)*(probability aguila given A) / (probability A)*(probability aguila) + (probability B)*(probability aguila)

theta_B_given_Aguila <- function(theta, theta_A, theta_B) {
  (1 - theta) * (1 - theta_B) / (theta * (1 - theta_A) + (1 - theta) * (1 - theta_B))
} # Bayes theorem - (probability b)*(probability aguila given B) / (probability A)*(probability aguila) + (probability B)*(probability aguila)

# EM Algorithm

for (i in 1:100) {
  
  # E-Step
  
  theta_A_6 <- theta_A_given_Aguila(theta, theta_A, theta_B) # probability of get a coin A in 6th trial
  theta_B_6 <- theta_B_given_Aguila(theta, theta_A, theta_B) # probability of get a coin B in 6th trial
  
  # M-Step
  
  theta <- (2 + theta_A_6) / 6 # update theta value
  theta_A <- (1 + 0) / (2 + theta_A_6) # update theta_a
  theta_B <- (1 + 0) / (3 + theta_B_6) # update theta_b
} 

# Estimated Values

cat(theta, theta_A, theta_B)

# The probability that coin A is selected is $38%$, so the probability that coin B is selected is $62%$.

# The probability that the result is Heads given that it is coin A is approximately $42%$.

# Finally, the probability that the result is Heads given that it is coin B is approximately $27%$.

cat(theta_A_6, theta_B_6)

# The probability that, in attempt no. 6, coin A was selected is $33%$, while the probability that coin B was selected is $66%$.

# Therefore, the probability that the missing data in attempt no. 6 is B is higher.
