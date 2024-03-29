# Given the dataset in the xlxs file obtain the simultaneous confidence intervals T-squared with 95% confidence level for every mean mu_{i} for i = (1, 2, 3, 4).

# Simultaneous confidence intervals given by

  # \bar{x_{p}} - \sqrt{\frac{(n - 1)p}{(n - 1)}F_{p, n - p}(\alpha)}\sqrt{\frac{S_{pp}}{n}} \leq \mu_{p} \leq \bar{x_{p}} + \sqrt{\frac{(n - 1)p}{(n - 1)}F_{p, n - p}(\alpha)}\sqrt{\frac{S_{pp}}{n}}

# Let a be \sqrt{\frac{(n - 1)p}{(n - 1)}F_{p, n - p}(\alpha)} and b be \sqrt{\frac{S_{pp}}{n}} \leq \mu_{p} \leq \bar{x_{p}} + \sqrt{\frac{(n - 1)p}{(n - 1)}F_{p, n - p}(\alpha)}\sqrt{\frac{S_{pp}}{n}}

library(dplyr) 
library(readxl)

data <- read_excel("C:/Users/Aspph/OneDrive/Desktop/MCE/EstadisticaMultivariada/Tareas/Tarea2/datos_osos.xlsx") # reading xlsx file
n <- nrow(data)
p <- ncol(data)

mean_vector <- colMeans(data)
print(mean_vector)

Fvalue <- 9.117 # value of Fisher distribution
#critical_value <- sqrt(((n - 1*p)/(n-1))*Fvalue) # a
critical_value <- sqrt((n-1)*p*(qf(0.95, p, n-p))/(n-p))

S <- cov(data) # covariance matrix S
S_updated <- matrix(sqrt(diag(S/n))) # b
print(S_updated)

lower_bound <- mean_vector - (critical_value*S_updated)
print(lower_bound) # lower bound
upper_bound <- mean_vector + (critical_value*S_updated)
print(upper_bound) # upper bound

# Obtain the simultaneous confidence intervals T-squared with 95% confidence level for the mean differences (annual increases).

a <- matrix(c(-1, 1, 0, 0, 0, -1, 1, 0, 0, 0, -1, 1), 4)

S_diff <- c()
for (i in 1:3){
  S_diff[i] <- sqrt((t(a[,i])%*%S%*%(a[,i]))/n) # variance differences
} 

S_diff <- as.vector(S_diff)

diff_mean_vector <- as.vector(diff(mean_vector))
print(diff_mean_vector) # differences between successive means (annual increases)

lower_bound_diff <- diff_mean_vector-critical_value*S_diff
print(lower_bound_diff) # lower bound
upper_bound_diff <- diff_mean_vector+critical_value*S_diff
print(upper_bound_diff) # upper bound




