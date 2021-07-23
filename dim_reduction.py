import math
import numpy as np
import time
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
import data
from matplotlib import pyplot, cm
from matplotlib.colors import Normalize


def main():

  X_strip = data.get_strip(n_samples=500, noise=0, length=10, width=5)
  X, y_true = data.roll(X_strip)

  # X, y_true = make_swiss_roll(n_samples=1000)
  # X_strip = np.stack((y_true, X[:, 1])).T


  n_neighbors = 10

  models = dict()
  models['isomap'] = {'model': Isomap(n_neighbors=n_neighbors, n_components=2)}
  models['lle'] = {'model': LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2)}
  models['ltsa'] = {'model': LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='ltsa')}

  for _, info in models.items():
    start = time.time()
    info['X'] = info['model'].fit_transform(X)
    info['time'] = round(time.time() - start, 3)

  n_plots = len(models) + 3
  plot_n_rows = 2
  plot_n_cols = math.ceil(n_plots / plot_n_rows)
  fig = pyplot.figure(figsize=(7 * plot_n_cols, 7 * plot_n_rows))

  y_true = Normalize()(y_true)
  cmv = cm.get_cmap('viridis')

  ax_roll = fig.add_subplot(plot_n_rows, plot_n_cols, 1, projection='3d')
  ax_roll.set_title('Roll')
  ax_roll.scatter(*[X[:, i] for i in range(X.shape[1])], c=y_true)

  ax_strip = fig.add_subplot(plot_n_rows, plot_n_cols, 2)
  ax_strip.set_title('Strip')
  # ax_strip.set_aspect('equal')
  ax_strip.scatter(*[X_strip[:, i] for i in range(X_strip.shape[1])], c=y_true)

  ax_graph = fig.add_subplot(plot_n_rows, plot_n_cols, 3)
  ax_graph.set_title('Graph')
  # ax_graph.set_aspect('equal')
  knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
  X_graph = knn.kneighbors_graph(X).toarray()
  # pyplot.imshow(X_graph, cmap='Greys')
  for i in range(X_graph.shape[0]):
    for j in range(i):
      if X_graph[i, j] + X_graph[j, i] > 0:
        ax_graph.plot(X_strip[[i, j], 0], X_strip[[i, j], 1], '-',
                      color=cmv((y_true[i] + y_true[j])/2))
  ax_graph.scatter(*[X_strip[:, i] for i in range(X_strip.shape[1])], c=y_true, s=1, linewidth=1)

  infos = list(models.items())

  for i in range(4, n_plots + 1):
    i_model = i - 4
    name, info = infos[i_model]
    ax_model = fig.add_subplot(plot_n_rows, plot_n_cols, i)
    ax_model.set_title('{}: {} secs'.format(name, info['time']))
    X_model = info['X']
    # ax_model.set_aspect('equal')
    ax_model.scatter(*[X_model[:, -i-1] for i in range(X_model.shape[1])], c=y_true)

  fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1)
  pyplot.show()


if __name__ == '__main__':
  main()
