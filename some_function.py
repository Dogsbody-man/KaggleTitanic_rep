def drop_features(dataset, column):
    if type(dataset) == list:
        for data in dataset:
            data.drop(column, axis=1, inplace=True)
        return dataset
    else:
        dataset.drop(column, axis=1, inplace=True)
        return dataset
    
def split_data_type(dataset, categorical_indices, numeric_indices):
    categorical_data = dataset[dataset.columns[categorical_indices]]
    numeric_data = dataset[dataset.columns[numeric_indices]]
    return categorical_data, numeric_data