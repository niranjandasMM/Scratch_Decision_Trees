import pandas as pd

def gini_impurity(df, target_column, feature_column):
    target_values = df[target_column].unique()
    feature_values = df[feature_column].unique()
    total_samples = len(df)

    impurity = 0
    for feature_value in feature_values:
        subset = df[df[feature_column] == feature_value]
        feature_impurity = 1
        for target_value in target_values:
            p = len(subset[subset[target_column] == target_value]) / len(subset)
            feature_impurity -= p ** 2
        impurity += (len(subset) / total_samples) * feature_impurity

    return impurity

def only_targt(target, feature_values):
    total_samples = len(target)
    gini_impurity = 1.0

    for value in feature_values:
        num_samples_with_value = len(target[target == value])
        proportion = num_samples_with_value / total_samples
        gini_impurity -= proportion ** 2

    return gini_impurity


def calculate_gini_impurity(dataset, target):
    gini_impurity_dict = {}
    target_column = target
    
    target_values = dataset[target_column].unique()

    # Calculate Gini impurity for the target variable
    target_gini_impurity = only_targt(dataset[target_column], target_values)
    gini_impurity_dict[target] = target_gini_impurity

    for feature in dataset.columns:
        if feature != target_column:
            gini_impurity_dict[feature] = gini_impurity(dataset, target_column, feature)

    return gini_impurity_dict


# gini_impurity_dict = calculate_gini_impurity(df, target='Play')
# print(gini_impurity_dict)
