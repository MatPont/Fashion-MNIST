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

times <- c()

#perpl <- 5

perpl_v <- c(1:20 * 5, 1:10 * 25 + 100)

save_plot = FALSE

#for(i in c(1:10)){
for(perpl in perpl_v){
  # Run tSNE
  start_time <- Sys.time()
  tsne <- Rtsne(train_x, perplexity = perpl, eta = 200.0, verbose = TRUE) 
  stop_time <- Sys.time()
  time_taken <- stop_time - start_time
  times <- c(times, time_taken)
  
  # Plot
  if(save_plot){
    name <- paste("tsne_", perpl, ".png", sep = "")
    colors = rainbow(length(unique(train_y)))
    names(colors) = unique(train_y)
    kl <- round(tsne$itercosts[length(tsne$itercosts)], digits = 2)
    plot_name <- paste("t-SNE with KL-divergence = ", kl, sep ="")
    png(filename=name, width = 1000, height = 1000)
    plot(tsne$Y, xlab = "Comp. 1", ylab = "Comp. 2", col = colors[train_y], main = plot_name)
    dev.off() 
  }
  
  # Increase perplexity
  #perpl <- perpl +5
}

print(times)

png(filename="tSNE_time", width = 1000, height = 1000)
plot(perpl_v, times, type = "b", xlab = "Perplexity", ylab = "Computation time")
dev.off() 

#kl_res <- c(4.09, 3.73, 3.48, 3.33, 3.18, 3.08, 2.99, 2.91, 2.83, 2.78, 2.73, 2.68, 2.62, 2.56, 2.54, 2.51, 2.47, 2.43, 2.39, 2.36, 2.21, 2.14, 2.03, 1.97, 1.88, 1.83, 1.77, 1.72, 1.67, 1.64)
#perpl_v <- c(1:20 * 5, 1:10 * 25 + 100)
#plot(perpl_v, kl_res, type = "b", xlab = "perplexity", ylab = "KL divergence")
