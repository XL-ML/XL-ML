# no parameters in this case, just {k}
defmodule KNN do
  defp get_all_predictions(class_to_num, y_train, sorted_distance_indexes, i, k) do
    if i == k do
      class_to_num
    else
      label = y_train[sorted_distance_indexes[i]]
      current_value = Map.get(class_to_num, label)
      if current_value == nil do
        get_all_predictions(Map.put(class_to_num, label, 1), y_train, sorted_distance_indexes, i + 1, k)
      else
        get_all_predictions(Map.put(class_to_num, label, current_value + 1), y_train, sorted_distance_indexes, i + 1, k)
      end
    end
  end

  defp prediction(x_train, y_train, sample, k) do
    euclidean_distances = Nx.sqrt(Nx.sum(Nx.power(Nx.subtract(x_train, sample), 2), axes: [:y]))

    # sort by distances (least to back) and track indices after sort
    sorted_distance_indexes = Nx.argsort(euclidean_distances)
    #IO.puts("sorted indexes")
    #IO.inspect(sorted_distance_indexes)

    class_to_num = get_all_predictions(%{}, y_train, sorted_distance_indexes, 0, k)

    #IO.puts("class to num")
    #IO.inspect(class_to_num)

    # find the maximum value in the map
    max = elem(Enum.max_by(Map.to_list(class_to_num), fn {k, v} -> v end), 0)

    max
  end

  defp iterate_predictions({x_train, y_train}, x_test, i, k, current_predictions) do
    if i == elem(Nx.shape(x_test), 0) do
      current_predictions
    else
      sample = x_test[i]
      prediction = prediction(x_train, y_train, sample, k)

      iterate_predictions({x_train, y_train}, x_test, i + 1, k, Nx.indexed_put(current_predictions, Nx.tensor([[i]]), Nx.tensor([Nx.to_number(prediction)])))
    end
  end

  def classify({x_train, y_train}, x_test, k) do
    n_test = elem(Nx.shape(x_test), 0)
    n_train = elem(Nx.shape(x_train), 0)
    if n_train < k do
      raise "k must be smaller than the number of train samples"
    end

    current_predictions = Nx.random_normal({n_test})
    iterate_predictions({x_train, y_train}, x_test, 0, k, current_predictions)

  end

  def metrics({x_train, y_train}, {x_test, y_test}, k) do
    predictions = classify({x_train, y_train}, x_test, k)

    amount_correct = Nx.sum(Nx.equal(predictions, y_test))
    Nx.divide(amount_correct, elem(Nx.shape(y_test), 0))

  end
end

# x_train = Nx.tensor([[1, 2], [2, 4], [3, 5], [4, 8]], names: [:x, :y])
# y_train = Nx.tensor([0, 0, 1, 1], names: [:x])
# x_test = Nx.tensor([[1, 2], [4, 7]], names: [:x, :y])
# y_test = Nx.tensor([0, 1], names: [:x])

# predictions = KNN.classify({x_train, y_train}, x_test, 3)
# #IO.inspect("predictions:")
# #IO.inspect(predictions)
# metric = KNN.metrics({x_train, y_train}, {x_test, y_test}, 3)
# #IO.inspect("metric:")
# #IO.inspect(metric)
