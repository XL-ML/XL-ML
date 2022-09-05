# XLML: Machine Learning for Elixir 

Elixir is a favorite among functional programmers and other enthusiasts. Support for native machine learning in Elixir, specifically regresssion algorithms, seems to be lacking for this erlang ~~clash of clans~~ based language. Through XL-ML, we bring ML algorithms, completely vectorized with [Nx](https://github.com/elixir-nx/nx) to the Elixir
community.

## Examples 
For most algorithms, all that is required is passing data and labels, getting a parameters object, and passing that in for future predictions and evaluation. 

An example is given below. All data is assumed to be of type `Nx.tensor`. 

```elixir
x = Nx.tensor([[1, 2], [2, 4]])
y =  Nx.tensor([2, 4])
```

After setting up data, we are ready to train. 

```elixir
params = Regressor.LinReg.fit(x, y, epochs: 1000, lr: 0.0001)
```

With these `params` we can compute a metric (r^2 for regressive tasks): 

```elixir 
r2_score = Regressor.LinReg.metric(x_test, y_test, params)
```

As you can see, XL-ML can do all of these things quickly and simply. 

## Algorithms 

A list of all algorithms in production is listed below. 
