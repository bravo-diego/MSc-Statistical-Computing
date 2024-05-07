# Binary (Presence-Absence) Similarity Coefficients

# "Coefficients which express relationships of either similarity of difference between units defined by binary data are especially important in systematic biology..." (Cheetham H. and Hazel E. (1969)).

# Given a set of 6 organisms (Lion, Giraffe, Cow, Sheep, Cat, Human), obtain the data matrix measuring 6 binary variables: X1 = tail, X2 = wild animal, X3 = long neck, X4 = farm animal, X5 = carnivorous, X4 = four-legged animal. 

library(arules)
library(nomclust)

data <- matrix(c(
  1, 1, 0, 0, 1, 1,
  1, 1, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1,
  1, 0, 0, 1, 0, 1,
  1, 0, 0, 0, 1, 1,
  0, 0, 0, 0, 1, 0
), nrow=6, byrow=TRUE) # data matrix

rownames(data) <- c("Lion", "Giraffe", "Cow", "Sheep", "Cat", "Human") # row names (organisms)
colnames(data) <- c("X1", "X2", "X3", "X4", "X5", "X6") # column names (characteristics)

# Obtain the distances matrix using the matching coefficients defined by a) Sokal-Michener; and b) Jaccard.

# Matching coefficient defined by Sokal and Michener (1958) gives the same weight to presents and absence of items.
dissimilarity(data, method = "matching") # compute and returns distances for binary data

# Distance measure by Jaccard method defined as the number of items which occur in both elements divided by the total number of items in the elements; this measure is often also called: binary, asymmetric binary, etc.
dissimilarity(data, method = "jaccard")
