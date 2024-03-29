# Linear Discriminant Analysis

# Given the wine data perform a Linear Discriminant Analysis (LDA), analyze the coefficients and plot the observations in the lower dimension space.

library(boot)
library(MASS)
library(caret)
library(DALEX)

file <- 'C:\\Users\\Aspph\\OneDrive\\Desktop\\MCE\\EstadisticaMultivariada\\Tareas\\Tarea3\\Supp Material\\wine.data'
data <- read.table(file, sep = ",") # read .dat file
colnames(data) = c('class','alcohol', 'malic', 'ash', 'alcal', 'mg', 'phenol', 'flav', 'nonf', 'proan', 'color', 'hue', 'abs', 'proline') # column names
data

lda <- lda(class~.,data = data)$x # discriminant functions LD1 and LD2

# Coefficients of the discriminant functions are useful for describing group differences and identifying variables that distinguish between groups. They represent the contribution of each variable to the classification of the groups in the space of the discriminant functions.

lda_coefficients <- lda$scaling # coefficients of the discriminant functions
lda_coefficients # variables that contribute most to distinguishing between wine classes are flav, nonf, abs, hue, and phenol; this is for the LD1 dimension. On the other hand, for the LD2 dimension, the variables ash and alcohol have a high coefficient compared to other variables, highlighting their relevance in this dimension

plot(lda, dimen = 2, main = "Linear Discriminant Analysis for Wine Dataset", xlab = "Discriminant Coordinate 1", ylab = "Discriminant Coordinate 2")

data$class <- as.character(data$class)

ctrl <- trainControl(method = "cv")  # cross-validation
model <- train(class ~ ., data = data, method = "lda", trControl = ctrl)

print(model)
