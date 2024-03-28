# Load the 'Maiz.csv' file and estimate the coefficients using the non-parametric kernel regression method.

  # See Nadaraya, E. A. (1964). ”On Estimating Regression”. Theory of Probability and its Applications. 9 (1): 141-2. doi:10.1137/1109020.

file <- "C:\\Users\\Aspph\\OneDrive\\Escritorio\\MCE\\InferenciaEstadistica\\Tareas\\Tarea4\\Maiz.csv"
df <- read.csv(file, fileEncoding = "Latin1", check.names = F)

str(df)  # Data frame with 200 obs. of 2 variables. Col names - ($P. Tonelada Maiz, $P. Tonelada Tortilla)

regression_curve <- function(x, h, vx, vy){
  n <- length(vx)
  numerator <- numeric(n)
  denominator <- numeric(n)
  for (i in 1:n){
    v <- (x - vx[i]) / h
    numerator[i] <- (vy[i] *  dnorm(v, mean = 0, sd = 1))
  }
  N <- sum(numerator)
  for (i in 1:n){
    w <- (x - vx[i]) / h
    denominator[i] <- dnorm(w, mean = 0, sd = 1)
  }
  W <- sum(denominator)
  return(N/W)
}

h = 3 # bandwidt h

yn <- c() # regression curve y bar
for (i in c(min(df$`P. Tonelada Maíz`):max(df$`P. Tonelada Maíz`))){
  yn <- c(yn, regression_curve(i, h, df$`P. Tonelada Maíz`, df$`P. Tonelada Tortilla`))
}

plot(df$`P. Tonelada Maíz`, df$`P. Tonelada Tortilla`, main = 'Non-parametric kernel regression', xlab = 'x', ylab = 'y', col = 'slategray', pch = 20)
lines(c(min(df$`P. Tonelada Maíz`):max(df$`P. Tonelada Maíz`)), yn, col = 'firebrick3') # adjusting model to data points
