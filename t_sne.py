import sys
from sklearn.manifold import TSNE
from mnist_reader import load_mnist

X_train, y_train = load_mnist('dataset', kind='train')

embedding, kl, n_iter = TSNE(perplexity = float(sys.argv[1]),
							early_exaggeration = float(sys.argv[2]),
							learning_rate = float(sys.argv[3]), verbose = 10).fit_transform(X_train)

print(kl)
