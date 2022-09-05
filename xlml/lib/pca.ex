defmodule PCA do
  import Nx.Defn

  # this corresponds to aggregate average reconstruction error
  def metric(z, u_truncated_t, x_centered) do
    elem_num = elem(Nx.shape(x_centered), 0) * elem(Nx.shape(x_centered), 1)

    x_reconstructed = Nx.dot(z, Nx.transpose(u_truncated_t))

    x_diff = Nx.subtract(x_centered, x_reconstructed)


    Nx.divide(Nx.sum(x_diff), elem_num)

  end

  # choose only the first gamma entries of U
  def fit(x, gamma) do

    k = elem(Nx.shape(x), 1)

    mu = Nx.mean(x, axes: [1])
    mu_square = Nx.outer(mu, Nx.transpose(mu))

    x_centered = Nx.subtract(Nx.transpose(x), Nx.broadcast(mu, Nx.transpose(x).shape))

    x_square = Nx.dot(x, Nx.transpose(x))


    cov_mat = Nx.subtract(Nx.divide(x_square, (k - 1)), Nx.multiply(mu_square, (k/(k - 1))))

    {u, s, v} = Nx.LinAlg.svd(cov_mat)



    u_t = Nx.transpose(u)


    u_truncated_t = Nx.slice(u_t, [0, 0], [elem(u_t.shape, 0), gamma])

    z = Nx.dot(x_centered, u_truncated_t)

    # ret value:
    {z, u_truncated_t, x_centered}
  end
end

x = Nx.tensor([[1,2],[3,4], [6,4]])
k = 2

{z, u_truncated_t, x_centered} = PCA.fit(x, k)

IO.inspect(z)


# m = PCA.metric(z, u_truncated_t, x_centered)

# IO.inspect(m) # 0 avg aggregate reconstruction error
