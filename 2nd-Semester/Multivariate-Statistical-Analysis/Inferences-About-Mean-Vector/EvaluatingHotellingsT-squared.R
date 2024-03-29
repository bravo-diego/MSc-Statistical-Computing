# Given the data matrix X evaluate Hotelling's T-squared to test H_{0}: mu_0 = [7, 11].

X = matrix(c(2, 8, 6, 8, 12, 9, 9, 10), nrow = 4) # matrix data
n <- nrow(X) # no. of rows

mu <- colMeans(X) # mean vector
print(mu)

mu_0 = matrix(c(7, 11), nrow = 1) # null hypothesis

P <- diag(n) - (1/n)*matrix(1, n, n) # P matrix; centering matrix plays an important role since we will use it to remove the column means from a matrix, centering the matrix
S <- t(X)%*%P%*%X/(n-1) # the variance-covariance matrix consists of the variances of the variables along the main diagonal and the covariances between each pair of variables in the other matrix positions
S_inverse <- solve(S)

Tsquared <- n*(mu - mu_0)%*%S_inverse%*%t(mu - mu_0) # Hotelling's T-squared

# Verify the normality of the data matrix X.

library(car) 
help("dataEllipse")

# The 'dataEllipse' function generates ellipses that represent the covariance structure of the data.

  # i) Calculates the covariance matrix S of the input data. This matrix describes the variability and the relationship between different variables in the data.
  
  # ii) Calculates the eigenvalues and eigenvectors of the covariance matrix S. The eigenvectors represent the directions of maximum variance in the data, and the eigenvalues represent the magnitude of variance along those directions.

  # iii) Calculates the angle of rotation and the lengths of the semi-major and semi-minor axes of the ellipse. These parameters determine the shape and orientation of the ellipse.

  # iv) Plots the ellipse, it uses the calculated parameters to draw an ellipse that represents the covariance structure of the data. The center of the ellipse is typically placed at the mean of the data.

dataEllipse(X[, 1], X[, 2], xlim = c(-7,20), ylim = c(3, 17), xlab = "", ylab = "", xaxt = "n", yaxt = "n", 
            pch = 19, col = c("dodgerblue", "Medium Orchid"), lty = 2, levels = c(0.50, 0.95), fill = TRUE, grid = FALSE) # levels parameter draw elliptical contours at these (normal) probability or confidence levels
mtext(side = 1, line = 2, at = 4, adj = 0, cex = 1.2, "Feature 1")
mtext(side = 2, line = 1.5, at = 9.5, adj = 0.5, cex = 1.2, "Feature 2")
mytitle = "Data Ellipse"
mysubtitle = "Elliptical Contours at 50 and 95% Confidence Level"
mtext(side=3, line=2, at=3.5, adj=0.001, cex=1.2, mytitle)
mtext(side=3, line=1, at=0.3, adj=0.15, cex=1, mysubtitle)

# Evaluate T-squared using its relationship with Wilks' Lambda.

mu_0 <- as.vector(mu_0) # null hypothesis
Sigma_estimated_mu0 <- (t(X) - mu_0)%*%t(t(X) - mu_0)
print(Sigma_estimated_mu0)

Sigma_estimated <- (t(X) - mu)%*%t(t(X) - mu)
print(Sigma_estimated)

Tsquared_Wilks <- (((n - 1)*det(Sigma_estimated_mu0))/det(Sigma_estimated)) - (n - 1)
print(Tsquared_Wilks)

# Evaluate the likelihood-ratio \Lambda and Wilks' lambda

det_Sigma_estimated <- det(Sigma_estimated)
print(det_Sigma_estimated)
det_Sigma_estimated_mu <- det(Sigma_estimated_mu0)
print(det_Sigma_estimated_mu)

Lambda = (det_Sigma_estimated/det_Sigma_estimated_mu)**(n - 2)
print(Lambda)

Wilks_lambda = det_Sigma_estimated/det_Sigma_estimated_mu
print(Wilks_lambda)
