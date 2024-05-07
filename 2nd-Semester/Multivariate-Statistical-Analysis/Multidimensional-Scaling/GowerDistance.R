# Gower's Distance 

# Gower's distance can be used to measure how different two records are. The records may contain combinations of logical, numerical, categorical or text data. The distance is always a number between 0 (identical) and 1 (maximally disimilar).

# The Gower distance is a similarity measure that can handle different types of variables, making it suitable for mixed datasets (containing different types of variables).

library(StatMatch)

data = data.frame(
  Player = c('Ronaldinho', 'Etoo', 'Xavi', 'Messi', 'Puyol', 'Raúl', 'Ronaldo', 'Beckham', 'Casillas', 'Cannavaro', 'Torres', 'Aguero', 'Maxi', 'Pablo', 'Maniche', 'Morientes', 'Joaquín', 'Villa', 'Ayala', 'Canizares', 'Jesús Navas', 'Puerta', 'Javi Navarro', 'Daniel Alves', 'Kanoute', 'Valeron', 'Arizmendi', 'Capdevila', ' Riki', 'Coloccini', 'Riquelme', 'Forlan', 'Cani', 'Javi Venta', 'Tachinardi', 'Pandiani', 'Tamudo', 'De la pena', 'Luis Garcia', 'Jonathan', 'Aimar', 'Diego milito', 'Savio', 'Sergio Garcia', 'Zapatear', 'Edu', 'Juanito', 'Melli', 'Capi', 'Doblas' ),
  X1 = c(15,21,6,7,1,7,18,4,0,0,24,14,10,3,3,13,5,22,1,0,2,6,7,2,12,9,8,3,7,2,10,17,4,0,4,6,10,2,8,4,6,9,3,7,5,6,2,5,7,0),
  X2 = c(26,25,26,19,28,29,30,31,25,33,22,18,25,25,29,30,25,24,33,36,20,21,32,23,29,31,22,28,26,24,28,27,25,30,31,30,28,30,25,21,26,27,32,23,21,27,30,22,29,25),
  X3 = c(1.78,1.8,1.7,1.69,1.78,1.8,1.83,1.8,1.85,1.76,1.83,1.72,1.8,1.92,1.73,1.86,1.79,1.75,1.77,1.81,1.7,1.83,1.82,1.71,1.92,1.84,1.92,1.81,1.86,1.82,1.82,1.72,1.75,1.8,1.87,1.84,1.77,1.69,1.8,1.8,1.68,1.81,1.71,1.76,1.73,1.82,1.83,1.81,1.75,1.84),
  X4 = c(71,75,68,67,78,73.5,82,67,70,75.5,70,68,79,80,69,79,75,69,75.5,78,60,74,75,64,82,71,78,79,80,78,75,75,69.5,73,80,74,74,69,68,72,60,78,68,69,70.5,74,80,78,73,78),
  X5 = c(1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0),
  X6 = c(2,3,5,1,5,5,2,9,5,4,5,1,1,5,8,5,5,5,1,5,5,5,5,2,6,5,5,5,5,1,1,7,5,5,4,7,5,5,5,5,1,1,2,5,5,2,5,5,5,5),
  X7 = c(2,2,4,3,3,3,1,3,4,2,4,3,3,4,2,3,4,3,1,3,3,3,3,2,1,3,3,4,3,2,2,3,3,3,4,1,3,3,3,3,2,2,2,3,3,3,4,3,2,3))

variables <- as.matrix(data[,c(2:8)]) # X variables

# Gower distance matrix represents the distances (dissimilarities) between players.

distances_matrix <- gower.dist(variables) # compute Gower's distance (dissimilarity) between units in a dataset or between observations in two distinct datasets

class_mds <- cmdscale(sqrt(distances_matrix), k = 2, eig = TRUE, 
                      add = FALSE, x.ret = TRUE) # classical multidimensional scaling (MDS); eigenvalues should be returned

config_players <- class_mds$points # matrix with up to k columns whose rows give the coordinates of the points chosen to represent the dissimilarities

players = c('Ronaldinho', 'Etoo', 'Xavi', 'Messi', 'Puyol', 'Raúl', 'Ronaldo', 'Beckham', 'Casillas', 'Cannavaro', 'Torres', 'Aguero', 'Maxi', 'Pablo', 'Maniche', 'Morientes', 'Joaquín', 'Villa', 'Ayala', 'Canizares', 'Jesús Navas', 'Puerta', 'Javi Navarro', 'Daniel Alves', 'Kanoute', 'Valeron', 'Arizmendi', 'Capdevila', ' Riki', 'Coloccini', 'Riquelme', 'Forlan', 'Cani', 'Javi Venta', 'Tachinardi', 'Pandiani', 'Tamudo', 'De la pena', 'Luis Garcia', 'Jonathan', 'Aimar', 'Diego milito', 'Savio', 'Sergio Garcia', 'Zapatear', 'Edu', 'Juanito', 'Melli', 'Capi', 'Doblas' )
dimnames(config_players)[[1]] <- players # add names to solution obtained by cmdscale function

plot(config_players[,1], config_players[,2], xlim = range(-0.5, 0.5), main = "Two-dimensional plot of 50 players from a classical MDS", ylim = range(-0.5, 0.5),
     xlab = "Dimension 1", ylab = "Dimension 2", type = "n", lwd = 2)
text(config_players[,1], config_players[,2],
     labels = abbreviate(row.names(config_players), minlength = 9), cex = 0.9, lwd = 2)

# Further research is required to understand which characteristics contributed most to the separation of certain groups. It is necessary to determine if players who are close to each other are similar in terms of physical attributes (age, height, weight, etc.), or if performance played a predominant role in the separation of these groups.
# This provides a visual perspective on the diversity in the dataset of players belonging to the Spanish league.
