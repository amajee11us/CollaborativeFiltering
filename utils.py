import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math

def getPerformanceScores(y_pred, y_true, type="classifier"):
    '''
    Implementation of RMSE and MAE scores for Collab Filtering
    and Accuracy, Precision, Recall, F1 score for Classifier
    '''
    if type == "classifier":
        conf_matrix = confusion_matrix(y_true, y_pred)

        precision = np.mean(np.diag(conf_matrix) / np.sum(conf_matrix, axis=0))

        recall = np.mean(np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))

        F1 = (2 * precision * recall) / (precision + recall)

        accuracy = accuracy_score(y_true, y_pred)

        return accuracy, precision, recall, F1
    else:
        diff = y_true - y_pred

        abs_error = np.abs(diff)
        mean_abs_error = np.mean(abs_error) # this is MAE

        diff_square = np.square(diff)
        mean_sq_error = np.mean(diff_square)
        rmse = np.sqrt(mean_sq_error)

        return mean_abs_error, rmse

def printResults(result_row, table_columns=['Mean Absolute Error', 'RMSE']):
    # print the results in a table
    tab = PrettyTable(table_columns)
    tab.add_row(np.round(result_row,3))
    print(tab)