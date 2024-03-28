# Load the 'Maiz.csv' file and estimate the coefficients of a simple linear regression.

file <- "C:\\Users\\Aspph\\OneDrive\\Escritorio\\MCE\\InferenciaEstadistica\\Tareas\\Tarea4\\Maiz.csv"
df <- read.csv(file, fileEncoding = "Latin1", check.names = F)

str(df)  # Data frame with 200 obs. of 2 variables. Col names - ($P. Tonelada Maiz, $P. Tonelada Tortilla)

model <- lm(df$`P. Tonelada Tortilla` ~ df$`P. Tonelada Maíz`, data = df)
summary(model) 

n <- length(df$`P. Tonelada Maíz`)

x <- sum(df$`P. Tonelada Maíz`) # sum of x
y <- sum(df$`P. Tonelada Tortilla`) # sum of y
x2 <- sum(df$`P. Tonelada Maíz`**2) # x squared
xy <- sum(df$`P. Tonelada Maíz` * df$`P. Tonelada Tortilla`) # sum of x times y

# Coefficients
m <-  ((n * xy) - (x * y)) / ((n * x2) - ((x)**2)) # slope m (B1)
b <- ((y)/n) - (m*(x/n)) # intercept b (B0)

# y = B0 + B1x 

plot(df$`P. Tonelada Maíz`, df$`P. Tonelada Tortilla`, type = 'p', main = 'Simple Linear Regression', xlab = 'x', ylab = 'y', col = 'steelblue', pch = 20)
abline(b, m, col = 'firebrick3') # adjusting model to data points
