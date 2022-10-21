import os
from symbol import parameters
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from parameterFactory import fetchMLPParameters, fetchKNNParameters, fetchSVCParameters, pickBestParams
from utils import getPerformanceScores
from sklearn.model_selection import RandomizedSearchCV
import math

# display results
from prettytable import PrettyTable

class DigitClassifier:
    def __init__(self, model_name='MLP', num_classes=10):
        self.model_name = model_name
        self.class_labels = np.array([x for x in range(num_classes)])

        '''
        Fetch the parameter iterator based on the model family
        Note : Each iterator returns a different list of parameters 
        '''
        if self.model_name == 'MLP':
            self.iterator, self.param_names = fetchMLPParameters()
        elif self.model_name == 'SVC':
            self.iterator, self.param_names = fetchSVCParameters()
        elif self.model_name == "KNN":
            self.iterator, self.param_names = fetchKNNParameters()
        else:
            print("[Error] Your choice of classifier does not exist.")
        
        self.pred_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        #column names to display
        self.table_columns = [x for x in self.param_names]
        self.table_columns.extend(self.pred_metrics)

    def __init_model(self, parameters):
        '''
        Generates a model based on two paramters:
        model_name : SVC/MLP/KNN
        parameters : choice of parameters passed to the model from iterator
        returns:
        An object of the model with the parameters baked into it
        '''
        if self.model_name == 'MLP':
            '''
            Parameters (in-order) : hidden_layer_sizes, activation, alpha (regterm), learning_rate, learning_rate_init, tol
            Assumption : Single hidden layer with variable number of neurons.
            '''
            self.classifier = MLPClassifier(hidden_layer_sizes=(parameters[0],),
                                            activation=parameters[1],
                                            alpha=parameters[2],
                                            learning_rate=parameters[3],
                                            learning_rate_init=parameters[4],
                                            max_iter=parameters[5],
                                            solver=parameters[6],
                                            tol=1e-7,
                                            verbose=True
                                            )
        elif self.model_name == 'SVC':
            self.classifier = SVC(C=parameters[0],
                                  kernel=parameters[1], 
                                  gamma=parameters[2],
                                  degree=parameters[3],
                                  tol=1e-7
                                  )
        elif self.model_name == "KNN":
            '''
            Parameters (in-order) : n_neighbors, weights, p, metric, n_jobs
            '''
            self.classifier = KNeighborsClassifier(n_neighbors=parameters[0], 
                                                   weights=parameters[1], 
                                                   p=parameters[2],
                                                   metric=parameters[3],
                                                   n_jobs=2
                                                   )
        else:
            print("[Error] Your choice of classifier does not exist.")

    def tune(self, X_train, Y_train, X_test, Y_test):
        '''
        Performs model tuning
        1. Choose set of parameters from iterator present in parameterFactory.py
        2. Initialize a new model with the parameters in step 1
        3. Run training and validation on the model
        4. Display a table of results from all possible model combination
        '''
        tab = PrettyTable(self.table_columns)
        best_params = [] # store the best performing model based on accuracy
        best_acc = 0.0
        exp_count = 1

        '''
        Initialize the classifier - The first value in the parameter list
        '''
        initial_params = pickBestParams(self.model_name)
        self.__init_model(initial_params)
        '''
        Initialize the Parameter Search - Greedy but randomized search
        '''
        print("---------------------------------------")
        print("Running Model Type : {}".format(self.model_name))
        param_searcher = RandomizedSearchCV(self.classifier, self.iterator, 
                                            n_iter = 10, random_state = 0, verbose = 10)
        '''
        Search for the best parameters using a Randomized Greedy Search
        '''
        search = param_searcher.fit(X_train, Y_train)
        
        print("Final Results of Ablation experiments ...")
        best_params = [search.best_params_[param_name] for param_name in self.param_names]
        predictions = search.predict(X_test)
        acc, pr, recall, f1 = getPerformanceScores(predictions, Y_test)
        best_params.extend(np.round([acc, pr, recall, f1],3))
        tab.add_row(best_params)
        print(tab)
        print("Done.")

        return best_params

    def trainval(self, X_train, Y_train, X_test, Y_test, params):
        '''
        Train and test a classifier with only a single parameter set
        Note: This is only called after the tuning process
        '''
        # Initialize the model with the passed params
        self.__init_model(params)
        # Train the model on the complete val set
        self.classifier.fit(X_train, Y_train)
        # Return the evaluated results on the complete test set
        return self.classifier.predict(X_test)
