import clustering
from data import *
from clustering import *


def main():
    data = load_data("london_sample_500.csv")
    data = add_new_columns(data)
    print("Part A:")
    data_analysis(data)
    print("Part B:")
    Ks = [2,3,5]
    transformed_data = transform_data(data, ["cnt","t1"])
    for k in Ks:
        print(f"k = {k}")
        labels, centroids = kmeans(transformed_data, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        path = rf"C:\Users\ariel\PycharmProjects\PythonProject\results_k{k}.png"
        visualize_results(transformed_data, labels, centroids, path)
        print()


if __name__ == '__main__':
    main()

