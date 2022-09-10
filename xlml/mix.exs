defmodule Xlml.MixProject do
  use Mix.Project

  def project do
    [
      app: :xlml,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      description: description,
      deps: deps(),
      package: package
    ]
  end

  defp description do
    """
    Native Machine Learning Algorithms for Elixir
    """
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:nx, "~> 0.2"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
     files: ["lib", "mix.exs", "README.md"],
     maintainers: ["sealion", "mlpp"],
     licenses: ["MIT"],
     links: %{"GitHub" => "https://github.com/XL-ML/XL-ML",}
     ]
  end
end
