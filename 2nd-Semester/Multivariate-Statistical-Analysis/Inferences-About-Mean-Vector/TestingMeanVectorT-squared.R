# Consider the data in the .xlsx file obtain the confidence interval given the data transformation

  # x_{1} = \sqrt[4]{mediciones ~de ~la ~radiacion ~con ~puerta ~cerrada}

  # x_{2} = \sqrt[4]{mediciones ~de ~la ~radiacion ~con ~puerta ~abierta}

library(dplyr) 
library(readxl)

data <- read_excel("C:/Users/Aspph/OneDrive/Desktop/MCE/EstadisticaMultivariada/Tareas/Tarea2/datos_radiacion.xlsx", 
                   col_names = c("closed_door", "open_door"), skip = 1) # reading xlsx file

data <- data%>%
  mutate(x1 = as.numeric(closed_door)**(1/4), x2 = as.numeric(open_door)**(1/4)) # data transformation x1 = \sqrt[4](closed door observations); x2 = \sqrt[4](open door observations)

mean_vector <- data %>% select(x1, x2) %>% colMeans() %>% as.vector()
print(mean_vector)

n <- nrow(data) # no. of rows
p <- ncol(data) - 2 # p value

S <- data %>% select(x1, x2) %>% cov() %>% as.matrix() # variance-covariance matrix
print(S)
S_inverse <- solve(S) # S^{-1} matrix
print(S_inverse)

# Evaluate the eigenvectors and eigenvalues in order to plot the data ellipse.

library(car) 
help("dataEllipse")

# The 'dataEllipse' function generates ellipses that represent the covariance structure of the data.

# i) Calculates the covariance matrix S of the input data. This matrix describes the variability and the relationship between different variables in the data.

# ii) Calculates the eigenvalues and eigenvectors of the covariance matrix S. 

# iii) Calculates the angle of rotation and the lengths of the semi-major and semi-minor axes of the ellipse. These parameters determine the shape and orientation of the ellipse.

# iv) Plots the ellipse, it uses the calculated parameters to draw an ellipse that represents the covariance structure of the data. The center of the ellipse is typically placed at the mean of the data.

mu <- c(0.562, 0.589) # vector mean
eigen_values <- eigen(S)$values # eigenvalues S matrix; represent the magnitude of variance along those directions.
print(eigen_values)
eigen_vectors <- eigen(S)$vectors # eigenvectors S matrix; represent the directions of maximum variance in the data
print(eigen_vectors)

dataEllipse(data$x1, data$x2, xlab = "", ylab = "", xaxt = "n", yaxt = "n",
              pch = 19, col = c("Steel Blue 3", "Sandy Brown"), lty = 2, levels = c(0.50, 0.95), fill = TRUE, grid = FALSE) # levels parameter draw elliptical contours at these (normal) probability or confidence levels
mtext(side = 1, line = 2, at = 0.55, adj = 0.5, cex = 1.2, "Feature 1")
mtext(side = 2, line = 1, at = 0.6, adj = 0.5, cex = 1.2, "Feature 2")
mytitle = "Data Ellipse"
mysubtitle = "Elliptical Contours at 50 and 95% Confidence Level"
mtext(side = 3, line = 2, at = 0.55, adj = 0.5, cex = 1.2, mytitle)
mtext(side = 3, line = 1, at = 0.55, adj = 0.5, cex = 1, mysubtitle)

# Evaluate Hotelling's T-squared to test H_{0}: mu_0 = [0.55, 0.60] with an alpha value 0.05.

points(mu[1], mu[2], pch = 19, col = "indianred2") # mu vector

X <- as.matrix(data %>% select(x1, x2))
mu_0 = matrix(c(0.55, 0.60), nrow = 1) # null hypothesis
mu_0 = as.vector(mu_0)

Tsquared <- n*(mean_vector - mu_0)%*%S_inverse%*%t(mean_vector - mu_0) # Hotelling's T-squared
print(Tsquared)

# Evaluate a = S^{-1}(\bar{x} - \mu_{0})

a <- S_inverse%*%t(mean_vector - mu_0)
print(a)

tsquared <- (n*(t(a)%*%(mean_vector - mu_0))**2)/t(a)%*%S%*%a
print(tsquared) # n o t e: t^{2} = T^{2}
