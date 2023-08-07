import pandas as pd
import math

def entropy(df, target_column, feature_column):
    target_values = df[target_column].unique()
    feature_values = df[feature_column].unique()
    total_samples = len(df)

    entropy = 0
    for feature_value in feature_values:
        subset = df[df[feature_column] == feature_value]
        feature_entropy = 0
        for target_value in target_values:
            p = len(subset[subset[target_column] == target_value]) / len(subset)
            if p != 0:
                feature_entropy -= p * math.log2(p)
        entropy += (len(subset) / total_samples) * feature_entropy

    return entropy

def entropy_for_target(target, feature_values):
    total_samples = len(target)
    entropy = 0

    for value in feature_values:
        num_samples_with_value = len(target[target == value])
        proportion = num_samples_with_value / total_samples
        if proportion != 0:
            entropy -= proportion * math.log2(proportion)

    return entropy

def calculate_entropy(dataset, target):
    entropy_dict = {}
    target_column = target
    
    target_values = dataset[target_column].unique()

    # Calculate entropy for the target variable
    target_entropy = entropy_for_target(dataset[target_column], target_values)
    entropy_dict[target] = target_entropy

    for feature in dataset.columns:
        if feature != target_column:
            entropy_dict[feature] = entropy(dataset, target_column, feature)

    return entropy_dict

# entropy_dict = calculate_entropy(df, target = 'Play')
# print(entropy_dict)
