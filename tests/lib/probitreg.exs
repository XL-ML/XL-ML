defmodule ProbitReg do
  import Nx.Defn

  defp probit(z) do # this is the intermediate value
    Nx.multiply(0.5, (Nx.add(1, Nx.erf(Nx.divide(z, Nx.sqrt(2))))))
  end
  def forward(x, params) do
    w = elem(params, 0)
    b = elem(params, 1)
    probit(Nx.add(Nx.dot(x, w), b))
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
    grad({w, b}, fn {w, b} -> cost(x, y, {w,b}) end)
  end

  defp update_recursion(t, maxTimes, x, y, w, b, lr) do
    if t < maxTimes do
      gradients = compute_grad(x, y, w, b)
      w_new = Nx.subtract(w, Nx.multiply(lr, elem(gradients, 0)))
      b_new = Nx.subtract(b, Nx.multiply(lr, elem(gradients, 1)))

      #IO.puts("Cost:")
      #IO.inspect(cost(x, y, {w_new, b_new}))

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

x = Nx.tensor([[1, 2], [2, 4]]) # {0, 1}, 0
y = Nx.tensor([0, 1])

params = ProbitReg.fit(x, y, 2000, 0.01)
#IO.inspect(ProbitReg.forward(x, params))
#IO.inspect(ProbitReg.metric(x, y, params))
