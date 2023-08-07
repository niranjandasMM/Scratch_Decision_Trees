
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from _entropy import calculate_entropy
from _gini import calculate_gini_impurity

data = pd.read_csv('cinema_shopping_tenis_decision.csv')
data


def split_based_on_gini_impurity(data, target):
    print("##############################")
    print("Data for Splitting:")
    print(data)

    lowest_impurity = calculate_gini_impurity(data, target)
    print(f"Lowest impurity of the data is: {lowest_impurity}")

    min_key, min_value = min(lowest_impurity.items(), key=lambda x: x[1])
    print(
        f"Next split based on feature: {min_key}, Gini Impurity: {min_value}")

    print("##############################")
    unique_values = data[min_key].unique()

    data_splits = {value: data[data[min_key] == value]
                   for value in unique_values}

    for i, split_df in enumerate(data_splits.values(), 1):
        print(f"Split {i} - Data:")
        print(split_df)

        unique_values = split_df['Decision'].unique()
        if len(unique_values) > 1:
            print(
                f"Split {i} - Multiple target values found, further splitting:")
            split_based_on_gini_impurity(split_df, target)
        else:
            print(
                f"Split {i} - Only one target value found: {unique_values[0]}")

    print("##############################")


split_based_on_gini_impurity(data=data, target='Decision')



