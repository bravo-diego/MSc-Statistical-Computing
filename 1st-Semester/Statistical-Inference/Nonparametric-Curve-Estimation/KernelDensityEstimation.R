# Kernel Density Estimate 

  # Non-parametric method to estimate the probability density function of a random variable based on kernels as weights.

# Write a function to estimate density using the KDE method.

kde <- function(x, h, data){ # parameters function - point of interest, h bandwidth, data set
  n <- length(data) 
  kernel_values <- numeric(n)
  
  for (i in 1:n){
    v <- (x - data[i]) / h
    kernel_values[i] <- dnorm(v, mean = 0, sd = 1) # Gaussian kernel 
  }
  
  estimate <- (sum(kernel_values))/(n*h)
  
  return(estimate)
}

# Load the 'Tratamientos.csv' file and estimate its density using the function created above for h values h = (20, 30, 60). Plot the density estimate. 

file <- "C:\\Users\\Aspph\\OneDrive\\Escritorio\\MCE\\InferenciaEstadistica\\Tareas\\Tarea4\\Tratamiento.csv"
df <- read.csv(file, header = FALSE)

dim(df) # dimension data frame - 86 rows, 1 column

df$V1 <- sort(df$V1)

  # Bandwidth value h = 20

densities <- c() # empty vector to store densities estimates
for (i in 1:length(df$V1)){
  densities <-  c(densities, kde(df$V1[i], 20, df$V1)) # Gaussian Kernel 
}

plot(df$V1, densities, type = 'l', main = 'Gaussian Kernel Density Estimate', xlab = 'Data points', ylab = 'Density estimation', col = 'steelblue2')
mtext(substitute(paste(italic("h = 20"))), side = 3)


  # Bandwidth value h = 30

densities <- c() # empty vector to store densities estimates
for (i in 1:length(df$V1)){
  densities <-  c(densities, kde(df$V1[i], 30, df$V1)) # Gaussian Kernel 
}

plot(df$V1, densities, type = 'l', main = 'Gaussian Kernel Density Estimate', xlab = 'Data points', ylab = 'Density estimation', col = 'orange')
mtext(substitute(paste(italic("h = 30"))), side = 3)


  # Bandwidth value h = 60

densities <- c() # empty vector to store densities estimates
for (i in 1:length(df$V1)){
  densities <-  c(densities, kde(df$V1[i], 60, df$V1)) # Gaussian Kernel 
}

plot(df$V1, densities, type = 'l', main = 'Gaussian Kernel Density Estimate', xlab = 'Data points', ylab = 'Density estimation', col = 'coral2')
mtext(substitute(paste(italic("h = 60"))), side = 3)




