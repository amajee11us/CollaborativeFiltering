import os
import numpy as np
import pandas as pd


class LoadNetflixData:
    def __init__(self, data_path, split = 'train'):
        self.data_path = data_path

        self.data_split = split

        # fetch raw data
        self.rawData = self.__readRawData(data_path)

        # create the data structure
        self.unique_movies = self.rawData.movieId.unique()
        self.unique_users = self.rawData.userId.unique()

        self.rawRatings = self.rawData.rating


    def __readRawData(self, filename):
        data = pd.read_csv(filename, 
                           names = ["movieId", 
                                    "userId", 
                                    "rating"]
                          )
        return data

    def getData(self, split='train'):
        pass