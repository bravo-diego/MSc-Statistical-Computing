# Perform a correspondence analysis of the crime data in the United States (file 'uscrime'). This dataset consists of 50 observations of 7 variables. The number of crimes in the year 1985 is reported for 50 states in the USA, classified according to 7 variables.

  # Determine the contributions for the first three dimensions. How can the third dimension be interpreted?

library(plotly)
library(dplyr)
library(ca)

path <- file.choose() # file path
uscrimes <- read.csv(path, header = FALSE, encoding = "latin1")
head(uscrimes) # data preview

uscrimes_cpy <- uscrimes # data set copy

rownames(uscrimes) <- uscrimes[, 1] # rename row names
uscrimes <- uscrimes[, -1] # delete column no. 1 

data_updated <- uscrimes[, 3:9] # update data; delete unnecessary columns
colnames(data_updated) <- c("Murder", "Rape", "Robbery", "Assault", "Burglary", "Larceny", "Auto Theft") # rename column names

# Three-dimension Correspondence Analysis 

help("ca") # performs CA including supplementary row and/or column points

corres <- ca(data_updated, nd = 3) # perform correspondence analysis with no. dimensions = 3

Cr <- corres$rowcoord # rows coordinates
Cc <-corres$colcoord  # columns coordinates

  # 3D Correspondence Analysis Plot by States

p = plot_ly() 
p = add_trace(p, x = Cr[,1], y = Cr[,2], z = Cr[,3],
              mode = 'text', text = rownames(Cr),
              textfont = list(color = "red"), showlegend = FALSE) 
p = add_trace(p, x = Cc[,1], y = Cc[,2], z = Cc[,3], 
              mode = "text", text = rownames(Cc), 
              textfont = list(color = "blue"), showlegend = FALSE) 
p <- config(p, displayModeBar = FALSE)
p <- layout(p, scene = list(xaxis = list(title = colnames(Cr)[1]),
                            yaxis = list(title = colnames(Cr)[2]),
                            zaxis = list(title = colnames(Cr)[3]),
                            aspectmode = "uscrimes"),
            margin = list(l = 0, r = 0, b = 0, t = 0))
p$sizingPolicy$browser$padding <- 0
my.3d.plot = p

  # 3D Correspondence Analysis Plot by Regions

region <- uscrimes[, 10] # vector containing no. region 

p = plot_ly() 
p = add_trace(p, x = Cr[,1], y = Cr[,2], z = Cr[,3],
              mode = 'text', text = region,
              textfont = list(color = "red"), showlegend = FALSE) 
p = add_trace(p, x = Cc[,1], y = Cc[,2], z = Cc[,3], 
              mode = "text", text = rownames(Cc), 
              textfont = list(color = "blue"), showlegend = FALSE) 
p <- config(p, displayModeBar = FALSE)
p <- layout(p, scene = list(xaxis = list(title = colnames(Cr)[1]),
                            yaxis = list(title = colnames(Cr)[2]),
                            zaxis = list(title = colnames(Cr)[3]),
                            aspectmode = "uscrimes"),
            margin = list(l = 0, r = 0, b = 0, t = 0))
p$sizingPolicy$browser$padding <- 0
my.3d.plot = p

# Grouping data by regions

uscrimes_cpy <- uscrimes_cpy %>%
  group_by(V11) %>%
  summarise(
    V3 = sum(V3),
    V4 = sum(V4),
    V5 = sum(V5),
    V6 = sum(V6),
    V7 = sum(V7),
    V8 = sum(V8),
    V9 = sum(V9),
    V11 = first(V11)
  )

uscrimes_cpy <- as.data.frame(uscrimes_cpy)

colnames(uscrimes_cpy) <- c("Region", "Murder", "Rape", "Robbery", "Assault", "Burglary", "Larceny", "Auto Theft") # rename column names
rownames(uscrimes_cpy) <- uscrimes_cpy[, 1] # rename row names
uscrimes_cpy <- uscrimes_cpy[, -1] # delete column no. 1 

corres <- ca(uscrimes_cpy, nd = 3) # perform correspondence analysis with no. dimensions = 3

Cr <- corres$rowcoord # rows coordinates
Cc <-corres$colcoord  # columns coordinates

  # 3D Correspondence Analysis Plot by States

p = plot_ly() 
p = add_trace(p, x = Cr[,1], y = Cr[,2], z = Cr[,3],
              mode = 'text', text = rownames(Cr),
              textfont = list(color = "red"), showlegend = FALSE) 
p = add_trace(p, x = Cc[,1], y = Cc[,2], z = Cc[,3], 
              mode = "text", text = rownames(Cc), 
              textfont = list(color = "blue"), showlegend = FALSE) 
p <- config(p, displayModeBar = FALSE)
p <- layout(p, scene = list(xaxis = list(title = colnames(Cr)[1]),
                            yaxis = list(title = colnames(Cr)[2]),
                            zaxis = list(title = colnames(Cr)[3]),
                            aspectmode = "uscrimes"),
            margin = list(l = 0, r = 0, b = 0, t = 0))
p$sizingPolicy$browser$padding <- 0
my.3d.plot = p

# Grouping the data by region we observe that the regions are distributed along the third dimension. Regions 1 and 4 are located at the lower end of the axis, while regions 2 and 3 are at the upper end.

  # A pattern is observed for the different types of crime. There is a greater separation of the variables along the third dimension.
