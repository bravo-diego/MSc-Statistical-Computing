# Classical Multidimensional Scaling (MDS)

# Bhattacharyya distance is a quantity which represents a notion of similarity between two probability distributions.

library(smacof)

population = c("French", "Czech", "Germanic", "Basque", "Chinese", "Ainu", 
               "Eskimo", "African-American", "Spanish", "Egyptian") # column names

proportions = c(0.21, 0.06, 0.06, 0.67, 0.25, 0.04, 0.14, 0.57, 0.22, 0.06,
                0.08, 0.64, 0.19, 0.04, 0.02, 0.75, 0.18, 0.00, 0.15, 0.67, 
                0.23, 0.00, 0.28, 0.49, 0.30, 0.00, 0.06, 0.64, 0.10, 0.06, 
                0.13, 0.71, 0.27, 0.04, 0.06, 0.63, 0.21, 0.05, 0.20, 0.54) # data points

genetic_proportions <- matrix(proportions, nrow = 10, byrow = T) # data matrix

# Bhattacharyya distance is also known in genetics as Cavalli Sforza distance given by 

  # d_{ij}^2 = arccos( sum_{l=1}^{k} sqrt{p_{il} p_{jl}})

cavalli_sforza_distances <- acos(sqrt(genetic_proportions)%*%sqrt(t(genetic_proportions)) ) # distance method with no biological assumptions 
cavalli_sforza_distances[is.na(cavalli_sforza_distances)] <- 0

help("cmdscale")

class_mds <- cmdscale(sqrt(cavalli_sforza_distances), k = 6, eig = TRUE, 
                      add = FALSE, x.ret = FALSE) # classical multidimensional scaling (MDS); eigenvalues should be returned

config_populations <- class_mds$points # matrix with up to k columns whose rows give the coordinates of the points chosen to represent the dissimilarities

cumulative_variance <- c() # empty vector

for(i in seq(1:9)){
  cumulative_variance[i] <- cmdscale(sqrt(cavalli_sforza_distances), k = i, eig = TRUE, 
                                     add = FALSE, x.ret = FALSE)$GOF[1] # GOF is a numeric vector of length 2 (g_1, g_2) where g_i = (sum_{j=1}^{k}lambda_j)(sum_{j=1}^{n}|lamda_j|) where lamnda_j are the eigenvalues (sorted in decreasing order)
}

plot(seq(1:9), cumulative_variance, type = 'o', main = 'Explained Variance PCoA',
     xlab = 'Eigenvalues', ylab = 'Cumulative Variance')

class_mds <- cmdscale(sqrt(cavalli_sforza_distances), k = 4, eig = TRUE, 
                      add = FALSE, x.ret = FALSE) # classical multidimensional scaling (MDS); eigenvalues should be returned

config_populations <- class_mds$points # matrix with up to k columns whose rows give the coordinates of the points chosen to represent the dissimilarities

populations <- c("French", "Czech", "Germanic", "Basque", "Chinese", "Ainu", "Eskimo", "African-American", "Spanish", "Egyptian") 
dimnames(config_populations)[[1]] <- populations # add names to solution obtained by cmdscale function

plot(config_populations[,1], config_populations[,2], xlim = range(-0.5, 0.5), main = "Two-dimensional plot of 10 populations from a classical MDS", ylim = range(-0.5, 0.5),
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2)
text(config_populations[,1], config_populations[,2],
     labels = abbreviate(row.names(config_populations), minlength = 9), cex = 0.9, lwd = 2)

# Representation obtained through the classical MDS model shows a clear separation between regions; representation clear and coherent.

# Multidimensional Scaling (MDS) based on stress minimization.

help("smacofSym")

# Ratio Type MDS

ratio_mds <- smacofSym(sqrt(cavalli_sforza_distances), ndim = 2, type = "ratio", init = "torgerson", itmax = 1000, eps = 1e-06) 
config_populations <- ratio_mds$conf

plot(config_populations[,1], config_populations[,2], main = "Two-dimensional plot of populations from a MDS Ratio type",
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2, xlim = range(-1, 1), ylim = range(-1, 1))
text(config_populations[,1], config_populations[,2],
     labels = abbreviate(population, minlength = 8), cex = 0.6, lwd = 2)

# Measure of goodness of fit in MDS is called 'stress'. It measures the diff between the observed (dis)similarity matrix; the lower the stress the better the fit.

stress_values <- c() # empty vector to store stress values
for (i in seq(1:9)){
  stress_values[i] <- smacofSym(sqrt(cavalli_sforza_distances), ndim = i, type = "ratio", init = "torgerson", itmax = 1000, eps = 1e-06)$stress
}

plot(seq(1:9), stress_values, type = "l", main = "Stress by Dimension Ratio Transformation",
     xlab = "Dimensions", ylab = "Stress") # using the elbow in the curve as a guide to the dimensionality of the data
lines(x = seq(1:10), y = rep(0.05, 10), type = "l", col = "red2")

# Interval Type MDS

interval_mds <- smacofSym(sqrt(cavalli_sforza_distances), ndim = 2, type = "interval", init = "torgerson", itmax = 1000, eps = 1e-06) 
config_populations <- interval_mds$conf

plot(config_populations[,1], config_populations[,2], main = "Two-dimensional plot of populations from a MDS Interval type",
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2, xlim = range(-1.1, 1.1), ylim = range(-1, 1))
text(config_populations[,1], config_populations[,2],
     labels = abbreviate(population, minlength = 8), cex = 0.6, lwd = 2)

stress_values <- c() 
for (i in seq(1:9)){
  stress_values[i] <- smacofSym(sqrt(cavalli_sforza_distances), ndim = i, type = "interval", init = "torgerson", itmax = 1000, eps = 1e-06)$stress
}

plot(seq(1:9), stress_values, type = "l", main = "Stress by Dimension Interval Transformation",
     xlab = "Dimensions", ylab = "Stress")
lines(x = seq(1:10), y = rep(0.05, 10), type = "l", col = "red2")

# Interval Type MDS

ordinal_mds <- smacofSym(sqrt(cavalli_sforza_distances), ndim = 2, type = "ordinal", init = "torgerson", itmax = 1000, eps = 1e-06) 
config_populations <- ordinal_mds$conf

plot(config_populations[,1], config_populations[,2], main = "Two-dimensional plot of populations from a MDS Ordinal type",
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2, xlim = range(-1.1, 1.1), ylim = range(-1, 1))
text(config_populations[,1], config_populations[,2],
     labels = abbreviate(population, minlength = 8), cex = 0.6, lwd = 2)

stress_values <- c() 
for (i in seq(1:9)){
  stress_values[i] <- smacofSym(sqrt(cavalli_sforza_distances), ndim = i, type = "ordinal", init = "torgerson", itmax = 1000, eps = 1e-06)$stress
}

plot(seq(1:9), stress_values, type = "l", main = "Stress by Dimension Ordinal Transformation",
     xlab = "Dimensions", ylab = "Stress")
lines(x = seq(1:10), y = rep(0.05, 10), type = "l", col = "red2")

# Clustering pattern observed remains consistent in the representations obtained both by the classical MDS model and the least squares approach.
