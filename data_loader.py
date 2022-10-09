import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.datasets import fetch_openml

'''
Part 1 - The Netflix Movies dataset.
This dataset contains movieIds and associated ratings provided by users.
'''
class LoadNetflixData:
    def __init__(self, data_path, split = 'train'):
        self.data_path = data_path

        self.data_split = split

        # fetch raw data
        self.rawData = self.__readRawData(data_path)

        # create the data structure
        self.unique_movies = list(self.rawData.movieId.unique())
        self.unique_users = list(self.rawData.userId.unique())

        self.rawRatings = self.rawData.rating.to_numpy()


    def __readRawData(self, filename):
        data = pd.read_csv(filename, 
                           names = ["movieId", 
                                    "userId", 
                                    "rating"]
                          )
        return data

    def getMoviesOfUser(self, userId):
        return self.rawData[self.rawData.userId == userId]

    def getUsersWhoRatedMovie(self, movieId):
        return self.rawData[self.rawData.movieId == movieId]

    def checkIfUserRatedMovie(self, movieId, userId):
        return userId in self.rawData[self.rawData.movieId == movieId].userId.unique()

    def getVoteData(self, split='train'):
        '''
        Compute the vote matrix for each movie
        out_dim : num_users x num_movies in the dataset
        '''
        self.vote_matrix = np.zeros((len(self.unique_users), len(self.unique_movies)))
        print("Calculating the vote matrix ...")
        self.vote_matrix = self.rawData.pivot_table(columns="movieId", index="userId", values="rating", fill_value=0.0).to_numpy()
        print("Created a vote matrix of dim - ", self.vote_matrix.shape)
        print("Done.")
        return self.vote_matrix

'''
Part 2 - MNIST digit classification dataset
'''
class MNISTloader:
    def __init__(self, dataset_name = 'mnist_784', num_classes=10):
        self.num_classes = num_classes

        # load dataset from Sklearn
        X, y = fetch_openml(dataset_name, version=1, return_X_y=True)

        # Scale dataset
        X = self.__scale_dataset(X)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.getData(X, y)

    def __scale_dataset(x):
        return x/255

    def getData(self, X, y):
        X_train = self.X[:60000]
        X_test = self.X[60000:]

        Y_train = self.y[:60000]
        Y_test = self.y[60000:]

        return X_train, Y_train, X_test, Y_test