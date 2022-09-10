![icon](icon.png)

# XLML: Machine Learning for Elixir 

Elixir is a favorite among functional programmers and other enthusiasts. Support for native machine learning in Elixir, specifically regresssion algorithms, seems to be lacking for this erlang based language. Through XL-ML, we bring ML algorithms, completely vectorized with [Nx](https://github.com/elixir-nx/nx) to the Elixir
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

With these `params` we can compute a metric: 

```elixir 
r2_score = Regressor.LinReg.metric(x_test, y_test, params)
```

The most intuitive metric is selected for each algorithm (e.g., r^2 for regression or reconstruction error for PCA).

As can be seen, XL-ML is efficient, optimized, and descriptive. 

## Algorithms 

A list of all algorithms in production is listed below. 

### Regression
- Linear
- Logistic
- Softmax
- Probit

### Clustering
- K-Nearest Neighbors Classification

### Dimensionality Reduction
- Principal Component Analysis (PCA)
