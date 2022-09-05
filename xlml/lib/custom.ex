defmodule Regressor.CustomReg do
  import Nx.Defn

  defp compute_grad(x, y, w, b, forward, metric, cost) do

    gradients = grad({w, b}, fn {w, b} -> cost.(x, y, {w, b}) end)
    {elem(gradients, 0), elem(gradients, 1)}
  end

  defp update_recursion(t, maxTimes, x, y, w, b, lr, forward, metric, cost) do
    if t < maxTimes do

      n = elem(Nx.shape(x), 0)
      gradients = compute_grad(x, y, w, b, forward, metric, cost)


      #IO.puts("Cost:")
      #IO.inspect(cost.(x, y, {w, b}))

      update_recursion(t + 1, maxTimes, x, y, Nx.subtract(w, Nx.multiply(lr, elem(gradients, 0))), Nx.add(b, Nx.multiply(lr, elem(gradients, 1))), lr, forward, metric, cost)
    else
      {w, b}
    end
  end

  def fit(x, y, epochs, lr, forward, metric, cost) do
    k = elem(Nx.shape(x), 1)

    w = Nx.random_normal({k, 1})
    b = Nx.random_normal({1})

    update_recursion(0, epochs, x, y, w, b, lr, forward,metric, cost)
  end
end

# x = Nx.tensor([[1, 2], [2, 4]], names: [:x, :y]) # {0, 1}, 0
# y = Nx.tensor([2, 4], names: [:x])
# Regressor.CustomReg.fit(x, y, 1000, 0.0001, &Regressor.LinReg.forward/2, &Regressor.LinReg.metric/3, &Regressor.LinReg.cost/3)
