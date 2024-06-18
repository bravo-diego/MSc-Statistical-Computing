# Linear Discriminant Analysis

# Linear Discriminant Analysis (LDA) is an approach used in supervised machine learning to solve multi-class classification problems. LDA separates multiple classes with multiple features through data dimensionality reduction.

  # LDA works by identifying a linear combination of features that separates or characterizes two or more classes of objects or events. LDA does this by projecting data with two or more dimensions into one dimension so that it can be more easily classified.

# Assumptions 

  # LDA approaches the problem by assuming that the conditional probability density functions p(x|y = 0) and p(x|y = 1) are both the normal distribution with mean and covariance parameters (mu_{0}, Sigma_{0}) and (mu_{1}, Sigma_{1}), respectively. 

  # Covariance matrices of the different classes are equal.

# Given the 'vaso' database perform a Linear Discriminant Analysis (LDA) and analyze the relationship between the volume and rate variables.

library(MASS)
library(caret)
library(ggplot2)
library(robustbase)

head(vaso)

plot(vaso$Volume, vaso$Rate, pch = 19, col = factor(vaso$Y), main = "Relationship Between Volume and Rate", xlab = "Volume", ylab = "Rate")

help("lda")
lda <- lda(Y ~ ., data = vaso)  
scores <- predict(lda, vaso)

# Coefficients of the discriminant functions are useful for describing group differences and identifying variables that distinguish between groups. They represent the contribution of each variable to the classification of the groups in the space of the discriminant functions.

print(lda$scaling) # coefficients of the discriminant functions

# Repeat the process by applying log in both variables.

plot(log(vaso$Volume), log(vaso$Rate), pch = 19, col = factor(vaso$Y), main = "Relationship Between Volume and Rate", xlab = "Volume", ylab = "Rate")
title(sub = "Logarithmic relationship between volume and rate", line = -18.0)

lda_log <- lda(Y ~ log(Volume)+log(Rate), data=vaso)
scores_log <- predict(lda_log, vaso)

print(lda_log$scaling) # coefficients of the discriminant functions
