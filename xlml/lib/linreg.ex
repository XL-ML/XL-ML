defmodule Regressor.LinReg do
  import Nx.Defn

  def forward(x, params) do
    w = elem(params, 0)
    b = elem(params, 1)
    Nx.add(Nx.dot(x, w), b)
  end

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

  def cost(x, y, params) do
    y_hat = forward(x, params)
    Nx.mean(Nx.power(Nx.subtract(y_hat, y), 2))
  end

  defp compute_grad(x, y, w, b) do
    grad({w, b}, fn {w, b} -> cost(x, y, {w,b}) end)
  end

  defp update_recursion(t, maxTimes, x, y, w, b, lr) do
    if t < maxTimes do
      gradients = compute_grad(x, y, w, b)
      w_new = Nx.subtract(w, Nx.multiply(lr, elem(gradients, 0)))
      b_new = Nx.subtract(b, Nx.multiply(lr, elem(gradients, 1)))

      update_recursion(t + 1, maxTimes, x, y, w_new, b_new, lr)
    else
      {w, b}
    end
  end

  def fit(x, y, epochs, lr) do
    w = Nx.random_normal({elem(Nx.shape(x), 1)})
    b = Nx.random_normal({1})

    params = update_recursion(0, epochs, x, y, w, b, lr)
    params
  end
end

# x = Nx.tensor([[1, 2], [2, 4]]) # {0, 1}, 0
# y = Nx.tensor([2, 4])

# params = Regressor.LinReg.fit(x, y, 20000, 0.00001)
# #IO.inspect(Regressor.LinReg.forward(x, params))
# #IO.inspect(Regressor.LinReg.metric(x, y, params))
