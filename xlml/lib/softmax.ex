defmodule Regressor.SoftmaxReg do
  import Nx.Defn

  def softmax(matrix_batch) do

    sum_vectors = Nx.sum(Nx.exp(matrix_batch), axes: [-1])

    {n, k} = Nx.shape(matrix_batch)

    Nx.divide(Nx.exp(matrix_batch), Nx.transpose(Nx.broadcast(sum_vectors, {k, n})))
  end

  def forward(x, params) do
    w = elem(params, 0)
    b = elem(params, 1)
    softmax(Nx.add(Nx.dot(x, w), b))
  end

  def metric(x, y, params) do
    y_hat = Nx.round(forward(x, params))
    amount_correct = Nx.sum(Nx.equal(y_hat, y))
    Nx.divide(amount_correct, elem(Nx.shape(y), 0))
  end

  def cost(x, y, params) do
    y_hat = forward(x, params)

    firstPart = Nx.multiply(y, Nx.log(Nx.add(1/1.0e8, y_hat)))
    Nx.divide(Nx.negate(Nx.sum(firstPart)), elem(Nx.shape(y), 0))
  end

  defp compute_grad(x, y, w, b) do

    gradients = grad({w, b}, fn {w, b} -> cost(x, y, {w, b}) end)
    {elem(gradients, 0), elem(gradients, 1)}
  end

  defp update_recursion(t, maxTimes, x, y, w, b, lr) do
    if t < maxTimes do

      n = elem(Nx.shape(x), 0)
      gradients = compute_grad(x, y, w, b)

      update_recursion(t + 1, maxTimes, x, y, Nx.add(w, Nx.multiply(lr, elem(gradients, 0))), Nx.add(b, Nx.multiply(lr, elem(gradients, 1))), lr)
    else
      {w, b}
    end
  end

  def fit(x, y, epochs, lr) do
    k = elem(Nx.shape(x), 1)
    num_classes = elem(Nx.shape(y), 1)

    w = Nx.random_normal({k, num_classes})
    b = Nx.random_normal({num_classes})

    update_recursion(0, epochs, x, y, w, b, lr)
  end
end

# x = Nx.tensor([[1, 2], [2, 4], [1, 1], [3, 4]], names: [:x, :y]) # {0, 1}, 0
# y = Nx.tensor([[0, 1], [1,0], [0, 1], [1, 0]], names: [:x, :y])

# params = Regressor.SoftmaxReg.fit(x, y, 50, 0.001)

# #IO.puts("Classification accuracy (training set)")
# #IO.inspect(Regressor.SoftmaxReg.metric(x, y, params))
