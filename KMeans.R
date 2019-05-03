setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/DataScience/Fashion-MNIST")

library(aricode)
library(R.matlab)
library(skmeans)

normalize <- function(x) {x / sqrt(rowSums(x^2))}
normalizeByCol <- function(df) { t( normalize( t(df) ) )}
sent_process <- function(x){ x[1] - x[2] + 1 }

# -------------- Dataset loading --------------
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


# ----------------------------------------
df <- read.csv("encoded_32_images.csv", row.names = 1)

dim(df)
mat_df <- as.matrix(df)
mat_df <- normalize(mat_df)
dim(mat_df)

#mat_df[(rowSums(mat_df) == 0), 1] = 1

# ----------------------------------------


#print("run PCA...")
#resPCA <- PCA(mat_df, scale.unit = FALSE, ncp = 50)

#write.csv(resPCA$ind$coord, "pca_coord.csv")
#write.csv(resPCA$eig, "pca_eig.csv")

#mat_df <- resPCA$ind$coord


library(infotheo)

v.measure <- function(a, b) {
  mi <- mutinformation(a, b)
  entropy.a <- entropy(a, a)$UV
  entropy.b <- entropy(b, b)$UV
  if (entropy.a == 0.0) {
    homogeneity <- 1.0
  } else {
    homogeneity <- mi / entropy.a
  }
  if (entropy.b == 0.0) {
    completeness <- 1.0
  } else {
    completeness <- mi / entropy.b
  }
  if (homogeneity + completeness == 0.0) {
    v.measure.score <- 0.0
  } else {
    v.measure.score <- (2.0 * homogeneity * completeness
                        / (homogeneity + completeness))
  }
  # Can also return homogeneity and completeness if wanted
  print(completeness)  
  print(homogeneity)
  print(v.measure.score)
  v.measure.score
}

# -------------- Run --------------

k <- 10

print("run kmeans...")
res_kmeans <- kmeans(mat_df, centers = k)
write.csv(res_kmeans$cluster, "doc2vec_kmeans_clusters.csv")

print("run spherical kmeans...")
res_skmeans <- skmeans(mat_df, k, control = list(verbose = TRUE), method = "pclust")
write.csv(res_skmeans$cluster, "doc2vec_skmeans_clusters.csv")

# ----------------------------------------
