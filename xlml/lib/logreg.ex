defmodule Regressor.LogReg do
  import Nx.Defn

  defp sigmoid(x) do
    Nx.divide(1.0, (Nx.add(1, Nx.exp(Nx.negate(x)))))
  end

  def forward(x, params) do
    w = elem(params, 0)
    b = elem(params, 1)
   sigmoid(Nx.add(Nx.dot(x, w), b))
  end

  def metric(x, y, params) do
    y_hat = Nx.round(forward(x, params))
    amount_correct = Nx.sum(Nx.equal(y_hat, y))
    Nx.divide(amount_correct, elem(Nx.shape(y), 0))
  end

  def cost(x, y, params) do
    y_hat = forward(x, params)
    firstPart = Nx.multiply(y, Nx.log(Nx.add(1/1.0e8, y_hat)))
    secondPart = Nx.multiply(Nx.subtract(1, y), Nx.log(Nx.add(1/1.0e8, Nx.subtract(1, y_hat))))
    Nx.negate(Nx.mean(Nx.add(firstPart, secondPart)))
  end

  defp compute_grad(x, y, w, b) do
    #IO.inspect(cost(x, y, {w, b}))
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
# y = Nx.tensor([0, 1])

# params = Regressor.LogReg.fit(x, y, 20000, 0.01)
# #IO.inspect(Regressor.LogReg.forward(x, params))
# #IO.inspect(Regressor.LogReg.metric(x, y, params))
