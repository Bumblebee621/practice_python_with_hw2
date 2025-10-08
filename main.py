from data import *
from clustering import *


def main():
        data = load_data("london_sample_500.csv")
        data = add_new_columns(data)
        data = data_analysis(data)


if __name__ == '__main__':
    main()

