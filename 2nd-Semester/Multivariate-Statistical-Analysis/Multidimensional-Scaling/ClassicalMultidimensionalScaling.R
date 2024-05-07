# Multidimensional Scaling

# Principal Coordinate Analysis (PCoA) is a multivariate analysis method that lets you analyze a proximity matrix, whether it's a dissimilarity matrix or a similarity matrix.

data <- matrix(c(
  1, 1, 0, 0, 1, 1,
  1, 1, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1,
  1, 0, 0, 1, 0, 1,
  1, 0, 0, 0, 1, 1,
  0, 0, 0, 0, 1, 0
), nrow=6, byrow=TRUE) # matrix data

rownames(data) <- c("Lion", "Giraffe", "Cow", "Sheep", "Cat", "Human") # row names (organisms)
colnames(data) <- c("X1", "X2", "X3", "X4", "X5", "X6") # column names (characteristics)

# Matching coefficient defined by Sokal and Michener (1958) gives the same weight to presents and absence of items.
dissimilarity(data, method = "matching") # compute and returns distances for binary data

help("cmdscale") # clasical multidimensional scaling (MDS) of a data matrix, also known as principal coordinates analysis

dmatrix_matching <- matrix(c(0, 0.68, 1.00, 1.00, 0.68, 1.00,
                             0.68, 0, 1.00, 1.00, 1.34, 1.68,
                             1.00, 1.00, 0, 0, 0.68, 1.34,
                             1.00, 1.00, 0, 0, 0.68,1.34,
                             0.68, 1.34, 0.68, 0.68, 0, 0.68,
                             1.00, 1.68, 1.34, 1.34, 0.68, 0),
                             nrow = 6, byrow = TRUE) # distances matrix from similarity matrix Sokal-Michener method

# Principal coordinates representation using the distance matrix obtained from the Sokal-Michener similarity coefficients.

class_mds <- cmdscale(sqrt(dmatrix_matching), k = 2, eig = TRUE, 
                         add = FALSE, x.ret = FALSE) # classical multidimensional scaling (MDS); eigenvalues should be returned

config_organisms <- class_mds$points # matrix with up to k columns whose rows give the coordinates of the points chosen to represent the dissimilarities

organisms <- c("Lion", "Giraffe", "Cow", "Sheep", "Cat", "Human") 
dimnames(config_organisms)[[1]] <- organisms # add names to solution obtained by cmdscale function

plot(config_organisms[,1], config_organisms[,2], xlim = range(-1, 1), main = "Two-dimensional plot of 6 organisms from a classical MDS", ylim = range(-1, 1),
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2)
text(config_organisms[,1],config_organisms[,2],
     labels = abbreviate(row.names(config_organisms), minlength = 9), cex = 0.9, lwd = 2)

B <- config_organisms%*%t(config_organisms) # B martix defined as B = XX'
b <- diag(B) 

r <- eigen(B) # spectral decomposition B = ULU'
U <- r$vectors # eigenvectors of B
L <- diag(r$values) # eigenvalues of B

# Adding 'elephant' to the data matrix; update similarity and distances matrix.

dmatrix_matching <- matrix(c(0, 0.68, 1.00, 1.00, 0.68, 1.00, 0.34,
                             0.68, 0, 1.00, 1.00, 1.34, 1.68, 0.34,
                             1.00, 1.00, 0, 0, 0.68, 1.34, 0.68,
                             1.00, 1.00, 0, 0, 0.68,1.34, 0.68,
                             0.68, 1.34, 0.68, 0.68, 0, 0.68, 0.68,
                             1.00, 1.68, 1.34, 1.34, 0.68, 0, 1.34,
                             0.34, 0.34, 0.68, 0.68, 0.68, 1.34, 0),
                           nrow = 7, byrow = TRUE) # distances matrix from similarity matrix Sokal-Michener method

distances_elephant <- c(0.34, 0.34, 0.68, 0.68, 0.68, 1.34, 0)

# Principal coordinates of a new organism given by x_{n + 1} = (1/2)Lambda^{-1}X'(b - d).

elephant_coordinates <- (solve(L[1:2, 1:2])/2)%*%(t(config_organisms))%*%(b - sqrt(distances_elephant[0:6])) # principal coordinates

config_organisms_updated <- matrix(c(config_organisms[,1], elephant_coordinates[1], config_organisms[,2], elephant_coordinates[2]), nrow = 7)

organisms_updated <- c("Lion", "Giraffe", "Cow", "Sheep", "Cat", "Human", "Elephant")
dimnames(config_organisms_updated)[[1]] <- organisms_updated # add names to solution obtained by cmdscale function

plot(config_organisms_updated[,1], config_organisms_updated[,2], xlim = range(-1, 1), main = "Two-dimensional plot of 7 organisms from a classical MDS", ylim = range(-1, 1),
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2)
text(config_organisms_updated[,1], config_organisms_updated[,2],
     labels = abbreviate(row.names(config_organisms_updated), minlength = 9), cex = 0.9, lwd = 2)

# There is a clear difference between wild and farm animals. Elephant is close to Giraffe and Lion as expected. 
