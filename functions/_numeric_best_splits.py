from  ._gini import calculate_gini_impurity  
from ._best_threshold import find_optimal_threshold

def numeric_best_splits(data, target):
    tree = {} 
    
    lowest_impurity = calculate_gini_impurity(data, target)
    min_key, min_value = min(lowest_impurity.items(), key=lambda x: x[1])
    
    tree['type'] = 'node'  
    tree['feature'] = min_key 
    
    threshold = find_optimal_threshold(data[min_key], data[target])
    tree['threshold'] = threshold 
    tree['subtrees'] = {} 
    
    for direction in ['left', 'right']:
        mask = data[min_key] <= threshold if direction == 'left' else data[min_key] > threshold
        subset_data = data[mask]
        
        unique_values = subset_data[target].unique()
        
        if len(unique_values) > 1:
            tree['subtrees'][direction] = numeric_best_splits(subset_data, target) 
        else:
            tree['subtrees'][direction] = {'type': 'leaf', 'prediction': unique_values[0]} 
        
    return tree