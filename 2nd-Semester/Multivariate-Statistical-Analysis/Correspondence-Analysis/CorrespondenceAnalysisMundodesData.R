# The dataset 'mundodes' represents 91 countries in which 6 variables have been observed: birth rate, death rate, infant mortality, life expectancy for men, life expectancy for women, and per capita GNP. From the dataset, the life expectancy for men and women has been taken. Four categories have been formed for both women and men. They are denoted as M1 and H1 for life expectancies between less than 41 years to 50 years, M2 and H2 for 51 to 60 years, M3 and H3 for 61 to 70 years, and M4 and H4 for between 71 to more than 80 years.

  # Perform a correspondence analysis of these data. What is the conclusion from the results of the simple correspondence analysis?

library("ca")
library("gplots")

mundodes <- matrix(c(10, 0, 0, 0,
                     7, 12, 0, 0,
                     0, 5, 15, 0,
                     0, 0, 23, 19), nrow = 4, byrow = T) # contingency table 

# Graphical display of contingency table

rownames <- c("M1", "M2", "M3", "M4") # row names
colnames <- c("H1", "H2", "H3", "H4") # column names

dimnames(mundodes) <- list(rownames, colnames)

mundodes_table <- as.table(mundodes) # convert data matrix as a table

balloonplot(t(mundodes_table), main = "mundodes", xlab = "", ylab = "", 
           label = FALSE, show.margins = FALSE) # each cell contains a dot whose size reflects the relative magnitude of the corresponding component

n <- sum(mundodes) # frequency

# Correspondence Analysis 

help("ca") # performs CA including supplementary row and/or column points

corres <- ca(mundodes, nd = 2) # two-dimension correspondence analysis 

Cr <- corres$rowcoord # rows coordinates
plot(Cr, xlim = range(-3, 3), ylim = range(-3, 3), main = "Row Points CA", xlab = "Dim1", ylab = "Dim2", lwd = 1, col = "dodgerblue1",  pch = 16)
text(Cr + 0.3, labels = c("M1", "M2", "M3", "M4"), col = 1, lwd = 2)
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

Cc <-corres$colcoord  # columns coordinates
plot(Cc, xlim = range(-3, 3), ylim = range(-3, 3), main = "Column Points CA", xlab = "Dim1", ylab = "Dim2", lwd = 1, col = "firebrick1", pch = 17)
text(Cc + 0.3, labels = c("H1", "H2", "H3", "H4"), col = 1, lwd = 4)
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

# Symmetric plot

  # Symmetric plot represents the row and column profiles simultaneously in a common space.

plot(Cr, xlim = range(-3, 3), ylim = range(-3, 3), main = "CA Symmetric Plot", xlab = "Dim1", ylab = "Dim2", lwd = 1, col = "dodgerblue1",  pch = 16)
points(Cc, col = "firebrick1", pch = 17)
text(Cr + 0.3, labels = c("M1", "M2", "M3", "M4"), col = 1, lwd = 2)
text(Cc + 0.3, labels = c("H1", "H2", "H3", "H4"), col = 1, lwd = 4)
abline(h = 0, lty = 2)
abline(v = 0, lty = 2) # both rows (blue points) and columns (red triangles) are represented in the same space using the principal coordinates

# Chi-square Test 

chisq <- chisq.test(mundodes) # reject null hypothesis; i.e. row and column variables are statistically significantly associated 

