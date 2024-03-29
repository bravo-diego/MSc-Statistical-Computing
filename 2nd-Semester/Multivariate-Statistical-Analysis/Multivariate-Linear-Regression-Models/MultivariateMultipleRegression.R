# Multivariate Multiple Regression

# Consider the problem of modeling the relationship between m responses y1, y2, ..., ym and a single set of predictor variables z1, z2, ..., zr.

  # In this case each response is assumed to follow its own regression model, i.e.

    # Y1 = B01 + B11z1 + ... + Br1zr + e1

    # Y2 = B02 + B12z1 + ... + Br2zr + e2

    # ...

    # Ym = B0m + B1mz1 + ... + Brmzr + em

# Given the data in the T1-5.dat file perform a linear regression model using a single outcome y1 = NO2 and both outcomes y1 = NO2 and y2 = O3. Analyze the residuals.

file <- 'C:\\Users\\Aspph\\OneDrive\\Desktop\\MCE\\EstadisticaMultivariada\\Tareas\\Tarea3\\Supp Material\\T1-5.dat'
data <- read.table(file) # read .dat file
colnames(data) = c('Wind', 'Solar_radiation', 'CO', 'NO', 'NO2', 'O3', 'HC') # column names
data

  # Fitting Linear Models

# lm is used to fit linear models, including multivariate ones. It can be used to carry out regression. 

# Multiple linear regression model using a single response (y1 = NO2)

help("lm")
regression_model <- lm(data$NO2 ~ data$Wind + data$Solar_radiation, data = data) # a typical model has the form response ~ terms where response is the (numeric) response vector and terms is a series of terms which specifies a linear predictor for response
# terms specification first + second indicates all the terms in first together with all the terms in second with duplicates removed
summary(regression_model)

residuals <- resid(regression_model) # residuals
plot(residuals, main = "Residuals") # residuals shows a clear pattern?
qqnorm(residuals) # residuals are normally distributed? 

# Multiple linear regression model using both responses (y1 = NO2; y2 = O3)

regression_model <- lm(cbind(data$NO2, data$O3) ~ data$Wind + data$Solar_radiation, data = data)
summary(regression_model) # each response follow its own regression model

residuals <- resid(regression_model) # residuals
plot(residuals[,1], main = "Residuals") 
qqnorm(residuals[,1]) 
plot(residuals[,2], main = "Residuals")
qqnorm(residuals[,2])

