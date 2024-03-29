# Multiple Linear Regression Analysis

# Extension of simple linear regression used to predict an outcome variable (y) on the basis of multiple distinct predictor variables z_{i}

  # Y = B0 + B1z1 + B2z2 + B3z3 + error

# Given n independent observations of the outcome variable Y

  # Y1 = B0 + B1z11 + B2z12 + ... + Brz1r + e1

  # Y2 = B0 + B1z21 + B2z22 + ... + Brz2r + e2

  # ...

  # Yn = B0 + B1zn1 + B2zn2 + ... + Brznr + en

# Thus the main objective of the Linear Regression Analysis is find the regression coefficients B and the error variance sigma^{2} that fit with the available data

# Given the data in the P1-4.dat file adjust a linear regression model with profits as the dependent variable and sales and assets as the independent variables

file <- 'C:\\Users\\Aspph\\OneDrive\\Desktop\\MCE\\EstadisticaMultivariada\\Tareas\\Tarea3\\Supp Material\\P1-4.DAT'
data <- read.table(file, header = FALSE, sep="\t") # read .dat file
print(data)

Sales <- c(108.28, 152.36, 95.04, 65.45, 62.97, 263.99, 265.19, 285.06, 92.01, 165.68) # sales billions
Profits <- c(17.05, 16.59, 10.91, 14.14, 9.52, 25.33, 18.54, 15.73, 8.10, 11.13) # profits billions
Assets <- c(1484.10, 750.33, 766.42, 1110.46, 1031.29, 195.26, 193.83, 191.11, 1175.16, 211.15) # assets billions

df <- data.frame(Sales, Assets, Profits)
n <- nrow(df)
p <- ncol(df)

# Fitting Linear Models

# lm is used to fit linear models, including multivariate ones. It can be ised to carry out regression. 

regression_model <- lm(Profits ~ Sales + Assets, data = df) # a typical model has the form response ~ terms where response is the (numeric) response vector and terms is a series of terms which specifies a linear predictor for response
# terms specification first + second indicates all the terms in first together with all the terms in second with duplicates removed
summary(regression_model)

beta <- coef(regression_model) # coefficients from the regression model (intercept and slope)
beta <- as.matrix(beta)

fitted(regression_model) # fitted values from the regression model

residuals <- resid(regression_model) # residuals (differences between observed and predicted values) from the regression model
residuals <- as.matrix(residuals)

sigma(regression_model) # residual standard error; estimate of variability of the residuals around the regression line

confint(regression_model, level = 0.95) # computes confidence intervals for one or more parameters in a fitted model 

plot(residuals, main = "Residuals")
qqnorm(residuals)

Z <- data.frame(Z = rep(1, n))
Z$V1 <- df$Sales
Z$V3 <- df$Assets

Z_matrix <- as.matrix(Z) # Z matrix 

sigma <- t(residuals)%*%residuals/(n-(p))
sigma_squared <- sigma[1][1] # sigma^{2}
var <- sigma_squared*solve(t(Z_matrix)%*%Z_matrix) # variance

f_value <- qf(0.95, p, n-p) # fisher distribution value

# Simultaneous Confidence Intervals

lower_bound <- beta-sqrt(diag(var))*sqrt(p*f_value)
lower_bound
upper_bound <- beta+sqrt(diag(var))*sqrt(p*f_value)
upper_bound

# Individual Confidence Intervals

t_value <- qt(0.975, df = n-p)

lower_bound <- beta-sqrt(diag(var))*sqrt(t_value)
lower_bound
upper_bound <- beta+sqrt(diag(var))*sqrt(t_value)
upper_bound





