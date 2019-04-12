setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/DataScience/Fashion-MNIST")

library("FactoMineR")
library("corrplot")
library("factoextra")

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
train_x = load_image_file("Dataset/train-images-idx3-ubyte")
test_x  = load_image_file("Dataset/t10k-images-idx3-ubyte")

# load labels
train_y = as.factor(load_label_file("Dataset/train-labels-idx1-ubyte"))
test_y = as.factor(load_label_file("Dataset/t10k-labels-idx1-ubyte"))

train_xy = cbind(train_x, train_y)
test_xy = cbind(test_x, test_y)

label_col = 785

resPCA <- PCA(train_xy, quali.sup = label_col, scale.unit = FALSE)
resPCA <- PCA(test_xy, quali.sup = label_col, scale.unit = FALSE)

layout(matrix(c(1,2), ncol=2))
plot.PCA(resPCA, choix = "ind", habillage = label_col, label = NULL)

var <- get_pca_var(resPCA)
corrplot(var$cos2)
corrplot(var$contrib, is.corr=FALSE)

fviz_pca_ind(resPCA, col.ind = train_xy[,label_col], label = "none", addEllipses = TRUE)
