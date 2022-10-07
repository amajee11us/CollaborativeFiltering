from data_loader import LoadNetflixData
from collab_filter import CollaborativeFiltering
from meters import *


if __name__ == "__main__":
    model = CollaborativeFiltering()

    data_path_train = "data/TrainingRatings.txt"
    data_path_test = "data/TestingRatings.txt"

    # load training data
    trainDataset = LoadNetflixData(data_path=data_path_train, split='train')

    # load test data
    testDataset = LoadNetflixData(data_path=data_path_test, split='test')

    model.train(trainDataset)

    predictions = model.inference(testDataset)

    # extract the GT ratings
    labels = testDataset.rawRatings

    # calculate metrics 
    getPerformanceScores(predictions, labels)