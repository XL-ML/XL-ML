defmodule PCA do
  import Nx.Defn

  def metric(x, y, params) do

    if elem(Nx.shape(y), 0) == 1 do
      -1.0
    else
      y_hat = forward(x, params)
      rss = Nx.sum(Nx.power(Nx.subtract(y, y_hat), 2))

      tss = Nx.sum(Nx.power(Nx.subtract(y, Nx.mean(y)), 2))

      corr = Nx.subtract(1, Nx.divide(rss, tss))
      corr
    end
  end

  def fit(x, k) do

    cov_mat = Nx.divide(Nx.dot(X, Nx.transpose(X))

    params
  end
end

x = Nx.tensor([[1, 2], [2, 4]]) # {0, 1}, 0
k = 1

params = PCA.fit(x, k)
