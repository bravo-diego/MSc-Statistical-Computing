# Correspondence Analysis

# Correspondence Analysis (CA) can be considered as an extension of Principal Component Analysis (PCA) designed for categorical rather than continuous data. It summarizes the data in a two-dimensional graphical format.

# CA is commonly applied to a contingency tables derived from two categorical variables.

  # For more than two categorical variables, Multiple Correspondence Analysis (MCA) should be applied.

# A sample of 901 people was cross-classified according to three income categories and four job satisfaction categories. The results are given in the following table. Perform a correspondence analysis of these data.

library("ca")
library("gplots")
library("ggplot2")

help("ca") # performs CA including supplementary row and/or column points

satisfaction_data <- matrix(c(42, 62, 184, 207,
                     13, 28, 81, 113,
                     7, 18, 54, 92), nrow = 3, byrow = T) # contingency table 

# Graphical display of contingency table

rownames <- c("< $25,000", "$25,000-$50,000", "> $50, 000") # row names
colnames <- c("Very dissatisfied", "Somewhat dissatisfied", "Moderately satisfied", "Very satisfied") # column names

dimnames(satisfaction_data) <- list(rownames, colnames)

satisfaction_data_table <- as.table(satisfaction_data) # convert data matrix as a table

balloonplot(t(satisfaction_data_table), main = "Job Satisfaction Data", xlab = "", ylab = "", 
            label = FALSE, show.margins = FALSE) # each cell contains a dot whose size reflects the relative magnitude of the corresponding component

  # One-dimension Correspondence Analysis 

corres <- ca(satisfaction_data, nd = 1) # one dimension included in the output

Cr <- corres$rowcoord # rows coordinates
Cc <-corres$colcoord  # columns coordinates

row_colors <- rep("dodgerblue1", length(Cr))
col_colors <- rep("firebrick1", length(Cc))

xlim <- range(c(Cr, Cc)) * 1.1

plot(Cr, rep(0, length(Cr)), col = row_colors, pch = 16, xlim = c(-2, 3), ylim = c(-1, 1), 
     xlab = "Dim 1", ylab = "", main = "CA Symmetric Plot", yaxt = "n", bty = "n")
points(Cc, rep(0, length(Cc)), col = col_colors, pch = 17)

text(Cr, rep(0.1, length(Cr)), labels = rownames, col = row_colors, pos = 3)
text(Cc, rep(-0.1, length(Cc)), labels = colnames, col = col_colors, pos = 1)
abline(h = 0, col = "black")
box() 

# Note: Associations between data cannot be fully appreciated.

  # Two-dimension Correspondence Analysis 

corres <- ca(satisfaction_data, nd = 2) # two dimensions included in the output

Cr <- corres$rowcoord # rows coordinates
Cc <-corres$colcoord  # columns coordinates

plot(Cr, xlim = range(-3, 3), ylim = range(-3, 3), main = "CA Symmetric Plot", xlab = "Dim1", ylab = "Dim2", lwd = 1, col = "dodgerblue1",  pch = 16)
points(Cc, col = "firebrick1", pch = 17)
text(Cr + 0.3, labels = c("< $25K", "$25K-$50K", "> $50K"), col = 1, lwd = 2)
text(Cc + 0.3, labels = c("Very dissatisfied", "Somewhat dissatisfied", "Moderately satisfied", "Very satisfied"), col = 1, lwd = 4)
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

# Note: 1st component divides the income categories; incomes greater than 25k shift to the left side of the graph

# Chi-square test of independence is used to analyze the frequency table formed by two categorical variables. 

  # Chi-square test evaluates whether there is a significant association between the categories of the two variables.

    # Null hypothesis (H0): the row and the column variables of the contingency table are independent.

    # Alternative hypothesis (H1): row and column variables are dependent.

chisq <- chisq.test(satisfaction_data) # reject null hypothesis; i.e. row and column variables are statistically significantly associated 
