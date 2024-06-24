# A company is evaluating the quality of its sales personnel by selecting a random sample of 50 salespersons. Each salesperson was assessed on three performance measures: sales growth, sales profitability, and new accounts sales. These measures have been converted to a scale where 100 indicates 'average' performance. Additionally, the 50 individuals were administered four tests designed to measure creativity, mechanical reasoning, abstract reasoning, and mathematical ability, respectively. The dataset consists of n = 50 observations on p = 7 variables, as shown in the file 'datosvendedores'.

  # Obtain the maximum likelihood solution for L and Psi for m = 2 and m = 3. Then calculate the communalities, specific variances, and LL'+ Psi for both solutions (i.e. m = 2 and m = 3 factors).

  # Test null hypothesis (H0: S = LL' + Psi) for m = 2 and m = 3. Which choice of m appears to be appropriate?

path <- file.choose() # file path

data <- read.csv(path, header = TRUE, encoding = "latin1")
head(data) # data preview

data_updated <- data[,-1]
data_updated <- scale(data_updated)
head(data_updated) # data preview

n <- 50 # observations
p <- 7 # variables
m <- 2 # factors

  # For m = 2

m = 2
estimation <- factanal(data_updated, m, rotation = "varimax")
loadings <- estimation$loadings
variances <- estimation$uniquenesses

L <-  matrix(c(
  0.852, 0.452,
  0.868, 0.419,
  0.717, 0.602,
  0.148, 0.987, 
  0.501, 0.525,
  0.619, 0,
  0.946, 0.277
), nrow =7 , byrow = TRUE) # loading matrix

Psi <- c(0.06919160, 0.07038038, 0.12330883, 0.00500000, 0.47358490, 0.61363862, 0.02881701)

S <- estimation$loadings%*%t(estimation$loadings)
diag(S) <- diag(S) + variances
print(diag(L%*%t(L)))

S <- L%*%t(L)
diag(S) <- diag(L%*%t(L)) + Psi
S

# Hypothesis testing

statistic <- estimation$STATISTIC
p_value <- estimation$PVAL # statistic p value

  # For m = 3

m = 3
estimation <- factanal(data_updated, m, rotation = "varimax")
loadings <- estimation$loadings
variances <- estimation$uniquenesses

L <-  matrix(c(
  0.793, 0.374 , 0.438,
  0.911, 0.317, 0.185,
  0.651, 0.544, 0.438,
  0.255, 0.964,   0   ,
  0.542, 0.465,  0.207,
  0.299,   0   , 0.950 ,
  0.917,  0.180, 0.298 
), nrow =7 , byrow = TRUE) # loading matrix

Psi <- c(0.03857165, 0.03448071, 0.08812176, 0.00500000, 0.44662048, 0.00500000, 0.03750980)

S <- estimation$loadings%*%t(estimation$loadings)
diag(S) <- diag(S) + variances
print(diag(L%*%t(L)))

S <- L%*%t(L)
diag(S) <- diag(L%*%t(L)) + Psi
S

  # Hypothesis testing

statistic <- estimation$STATISTIC 
p_value <- estimation$PVAL # statistic p value

# Calculate the factor scores using 1) weighted least squares, and 2) regression approach.

regression <- factanal(data_updated, factors = 3, scores = "regression")
scores <- regression$scores
scores[1,]

least_squares <- factanal(data_updated, factors = 3, scores = "Bartlett")
scores <- least_squares$scores
scores[1,]

