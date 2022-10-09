from data_loader import LoadNetflixData
from collab_filter import CollaborativeFiltering
from utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark for comparing Submodular Functions against Continuous Learning functions')

    # General parser
    parser.add_argument('-g',
                        '--question',
                        dest='dataset_format',
                        type=int,
                        default=1,
                        help='The question number of the problem.')
    parser.add_argument('--dataset_name',
                        default='Netflix',
                        type=str,
                        help='Name of the dataset. Example : enron1. The data is stored in the data directory')
    parser.add_argument('--model_name',
                        default='SVC',
                        type=str,
                        help='For question 2 only. Name of the model for which experiment is triggered. Example : SVC')
    parser.add_argument('--num_iters',
                        default=1000,
                        type=int,
                        help='Number of iterations to train your model for. Enter a value between 0 to 50k.')
    parser.add_argument('--learning_rate',
                        default=0.01,
                        type=float,
                        help='The initial learning rate for the model. Set it between 0 and 1.')
    parser.add_argument('--reg_const',
                        default=0.5,
                        type=float,
                        help='The regularization constant for the model. Set it between 0 and 1.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.question == 1:
        '''
        Collaborative filtering problem
        '''
        data_path_train = "data/TrainingRatings.txt"
        data_path_test = "data/TestingRatings.txt"

        # load training data
        trainDataset = LoadNetflixData(data_path=data_path_train, split='train')

        # load test data
        testDataset = LoadNetflixData(data_path=data_path_test, split='test')

        # Total users in the system
        total_num_users = len(trainDataset.unique_users)
        print("Total Number of users in the system : {}".format(total_num_users))

        # create model
        model = CollaborativeFiltering(num_users=total_num_users)

        model.train(trainDataset)

        print("Running Inference on : {} datapoints ...".format(testDataset.rawRatings.shape[0]))
        predictions = model.inference(testDataset)
        print("Done.")

        # extract the GT ratings
        labels = testDataset.rawRatings

        # calculate and display metrics 
        mae, rmse = getPerformanceScores(predictions, labels)

        # display the results
        print("Results from the experiment -->")
        printResults(result_row=[mae,rmse], table_columns=['Mean Absolute Error', 'RMSE'])
    else:
        '''
        Execution of question2 : SVC, MLP and k-NN problem
        '''
