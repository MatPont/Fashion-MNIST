setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/DataScience/Fashion-MNIST")

library("Rtsne")

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

perpl <- 5
for(i in c(1:10)){
  # Run tSNE
  tsne <- Rtsne(train_x, perplexity = perpl, eta = 200.0) 
  
  # Plot
  name <- paste("tsne_", perpl, ".png", sep = "")
  colors = rainbow(length(unique(train_y)))
  names(colors) = unique(train_y)
  kl <- round(tsne$itercosts[length(tsne$itercosts)], digits = 2)
  plot_name <- paste("t-SNE with KL-divergence = ", kl, sep ="")
  png(filename=name, width = 1000, height = 1000)
  plot(tsne$Y, xlab = "Comp. 1", ylab = "Comp. 2", col = colors[train_y], main = plot_name)
  dev.off()
  
  # Increase perplexity
  perpl <- perpl +5
}

