from data_loader import LoadNetflixData, MNISTloader
from collab_filter import CollaborativeFiltering
from classifier import DigitClassifier
from utils import *
import argparse
from parameterFactory import pickBestParams

def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark for comparing Submodular Functions against Continuous Learning functions')

    # General parser
    parser.add_argument('-q',
                        '--question',
                        dest='question',
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
    parser.add_argument('--no_tuning',
                        default=False,
                        type=bool,
                        help='Choose to perform or ignore tuning the model. Only applicable for question 2')
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
        mae, rmse = getPerformanceScores(predictions, labels, "collab")

        # display the results
        print("Results from the experiment -->")
        printResults(result_row=[mae,rmse], table_columns=['Mean Absolute Error', 'RMSE'])
    else:
        '''
        Execution of question2 : SVC, MLP and k-NN problem
        '''

        # Data loader class for MNIST dataset 
        print("Loading MNIST dataset ...")
        data_loader = MNISTloader(dataset_name = args.dataset_name, num_classes=10)
        print("Done.")
        
        print("Loading Model ...")
        model = DigitClassifier(model_name=args.model_name, num_classes=10)
        print("Done.")

        if args.no_tuning:
            print("Starting Model Trainval ...")
            final_params = pickBestParams(args.model_name)
        else:
            print("Starting Model Tuning ...")
            final_params = model.tune(data_loader.X_train, data_loader.Y_train, 
                                      data_loader.X_test, data_loader.Y_test)
            print("Done.")
        predictions = model.trainval(data_loader.X_train, data_loader.Y_train, 
                                      data_loader.X_test, data_loader.Y_test,
                                      final_params)
        print("Done.")

        metrics = getPerformanceScores(predictions, data_loader.Y_test, "classifier")
        printResults(result_row=[x for x in metrics], table_columns=model.pred_metrics)
        