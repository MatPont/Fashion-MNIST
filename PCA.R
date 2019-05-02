setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/DataScience/Fashion-MNIST")

library("FactoMineR")
library("corrplot")
library("factoextra")
library("fields")

# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
train_x = load_image_file("dataset/train-images-idx3-ubyte")
test_x  = load_image_file("dataset/t10k-images-idx3-ubyte")

# load labels
train_y = as.factor(load_label_file("dataset/train-labels-idx1-ubyte"))
test_y = as.factor(load_label_file("dataset/t10k-labels-idx1-ubyte"))

train_xy = cbind(train_x, train_y)
test_xy = cbind(test_x, test_y)

label_col = 785

# Compute PCA
resPCA <- PCA(train_xy, quali.sup = label_col, scale.unit = FALSE, ncp = 50)
resPCA <- PCA(test_xy, quali.sup = label_col, scale.unit = FALSE, ncp = 50)

layout(matrix(c(1,2), ncol=2))
plot.PCA(resPCA, choix = "ind", habillage = label_col, label = NULL)

#var <- get_pca_var(resPCA)
#corrplot(var$cos2)
#corrplot(var$contrib, is.corr=FALSE)

fviz_pca_ind(resPCA, col.ind = train_xy[,label_col], label = "none", addEllipses = TRUE)

# Images reconstruction
rec <- reconst(resPCA, ncp = 50)

# Select one obs for each class
index <- c(2, 17, 6, 4, 20, 9, 19, 7, 24, 1)
rec <- rec[index,]

normalize <- function(x){
  (x - min(x)) / (max(x) - min(x)) * 255
}

rec <- read.csv("decoded_images.csv", row.names = 1)

rec <- apply(rec, MARGIN = 1, FUN = normalize)

colors <- gray.colors(255)

for(i in 1:length(index)){
  layout(matrix(c(1:2), ncol=2))
  temp <- matrix(rec[,i], nrow = 28, byrow = TRUE)
  image(temp, col = colors)
  temp <- matrix(as.double(train_x[index[i],]), nrow = 28, byrow = TRUE)
  image(temp, col = colors) 
  readline(prompt="Press [enter] to continue")
}

temp <- matrix(rec[,1], nrow = 28, byrow = TRUE)
image.plot(temp, col = colors)
temp <- matrix(as.double(train_x[index[1],]), nrow = 28, byrow = TRUE)
image.plot(temp, col = colors) 
