import numpy as np
from prettytable import PrettyTable
import math

def getPerformanceScores(y_pred, y_true):
    '''
    Implementation of RMSE and MAE scores 
    '''
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