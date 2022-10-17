from itertools import product

def fetchMLPParameters():
    params = {
        'hidden_layer_sizes': [10, 100, 1000],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha' : [0.0001, 0.01, 0.1, 1.0],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter' : [10, 100, 1000],
        'solver': ['sgd', 'adam']
    }

    param_list = list(params.keys())

    return params, param_list

def fetchSVCParameters():
    params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma' : [0.01, 0.1, 1, 10, 100],
        'degree': [3, 5]
    }

    param_list = list(params.keys())

    return params, param_list

def fetchKNNParameters():
    params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p' : [1,2,3],
        'metric': ['minkowski', 'cosine', 'euclidean']
    }

    param_list = list(params.keys())

    return params, param_list

def pickBestParams(model_name):
    '''
    This is used either to initialize the params and to store the best ones.
    The order is determined based on the order of params above.
    '''
    if model_name == "MLP":
        return [100, 'logistic', 0.1, 'constant', 0.01, 100, 'sgd']
    elif model_name == "SVC":
        return [0.01, 'linear', 0.1, 3]
    elif model_name == "KNN":
        return [3, 'distance', 2, 'euclidean']
    else:
        return [] # return a blank array as the options dont match
