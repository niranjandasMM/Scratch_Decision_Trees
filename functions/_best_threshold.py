from  ._gini import calculate_gini_impurity  

def gini_impurity(labels):
    total_samples = len(labels)
    class_counts = labels.value_counts()
    gini = 1.0

    for count in class_counts:
        p_i = count / total_samples
        gini -= p_i ** 2

    return gini

def find_optimal_threshold(feature_values, labels):
    best_gini = float('inf')
    best_threshold = None

    for threshold in feature_values:
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold

        left_gini = gini_impurity(labels[left_mask])
        right_gini = gini_impurity(labels[right_mask])
        weighted_gini = (left_gini * sum(left_mask) + right_gini * sum(right_mask)) / len(labels)
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_threshold = threshold

    return best_threshold
