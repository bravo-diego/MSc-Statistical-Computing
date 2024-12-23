# Data Imputation using a Regression-Based Method

x1 <- c(1.43, 1.62 ,2.46, 2.48, 2.97, 4.03, 4.47, 5.76, 6.61, 6.68, 6.79, 7.46, 7.88, 8.92, 9.42)
x2 <- c(NA, NA, NA, -5.20, -6.39, 2.87, -7.88, -3.97, 2.32, -3.24, -3.56, 1.61, -1.87, -6.60, -7.64) # data from a normal bivariate distribution

df <- data.frame(x1, x2)
print(df)

N <- 3
up_data <- tail(df, -3) # drop missing values

mean_x1 <- mean(up_data$x1)
mean_x1

mean_x2 <- mean(up_data$x2)
mean_x2

cov_11 <- cov(up_data$x1, up_data$x1)
cov_11

cov_12 <- cov(up_data$x1, up_data$x2)
cov_12

# Regression Equation

  # E(X2 | X1) = \bar{X_2} + S12/S11 (x1 - \bar{x1})

Imputed_x2 <- ifelse(is.na(df$x2), mean_x2 + (cov_12 / cov_11) * (df$x1 - mean_x1), df$x2) # regression equation

df$Imputed_x2 <- Imputed_x2
df

# Original data points were: -0.69, -5.00, -1.13
