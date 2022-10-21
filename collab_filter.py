import pandas as pd
import numpy as np
from tqdm import tqdm

class CollaborativeFiltering:

    def __init__(self, num_users, metric="correlation"):
        self.mean_votes = pd.DataFrame(columns = ['userId', 'meanRating'])
        self.weight = np.zeros((num_users, num_users)) #pd.DataFrame(columns = ['activeUserId', 'otherUserId', 'weight']) 
        self.metric = metric
        self.num_users = num_users
        # calculate the normalizing constant
        self.k = (1/ num_users)
        self.users = [] # this is a map of user to key
        self.movies = []

    def calculateWeightMatrix(self, data_matrix):
        data_matrix = (data_matrix.T - self.mean_votes.meanRating.to_numpy()).T

        p_numerator  = np.matmul(data_matrix, data_matrix.T)
        sum_square_per_user = np.sum(np.square(data_matrix), axis=1)
        
        p_denominator = np.sqrt(np.matmul(sum_square_per_user, sum_square_per_user.T))

        corr_coeff = p_numerator / p_denominator
        # coefficients on the diagonal compare user a with itself and thus should be 1
        np.fill_diagonal(corr_coeff, 1.0)
        
        return corr_coeff

    
    def train(self, D_train : object):
        '''
        The training objective is to calculate the weights and mean votes
        '''
        # map users 
        self.users = D_train.unique_users
        self.movies = D_train.unique_movies

        # compute the mean vote per user
        print("Calculating Mean Vote Per User ...")
        for user in tqdm(D_train.unique_users):
            mean_vote_per_user = {
                'userId': int(user), 
                'meanRating': D_train.getMoviesOfUser(user).rating.mean()
            }
            
            if self.mean_votes.empty:
                self.mean_votes = self.mean_votes.append(mean_vote_per_user, ignore_index=True)
            else:
                mean_vote_per_user = pd.Series(mean_vote_per_user)
                self.mean_votes = pd.concat([self.mean_votes, mean_vote_per_user.to_frame().T], ignore_index=True)
        print("Done.")

        # Fetch the voting matrix and compute the weight matrix
        self.vote_matrix = D_train.getVoteData()
        # compute the weights w(a,i)
        print("Calculating Weight between active users and all other users ...")
        self.weight = self.calculateWeightMatrix(self.vote_matrix)

        print("Done.")

    def inference(self, D_test : object):
        test_data = D_test.rawData.to_numpy()

        predicted_vote = []
        true_vote = []

        for test_item in tqdm(test_data):
            a = test_item[1]
            item_j = test_item[0]
            # estimate the mean vote of the active user -- from training data
            vote_a_avg = self.mean_votes.loc[self.mean_votes.userId == a, 'meanRating'].values[0]
            weight_vector_a = self.weight[self.users.index(a)]
            vote_vector_j = (self.vote_matrix[:, self.movies.index(item_j)].T - self.mean_votes.meanRating.to_numpy()).T

            # find the sum of votes - mean for all other users -- from training data
            weighted_vote_vector = np.dot(weight_vector_a, vote_vector_j)  
            vote_a_j = vote_a_avg + self.k * np.sum(weighted_vote_vector)

            predicted_vote.append(vote_a_j)
            true_vote.append(test_item[2])
        
        return np.array(predicted_vote)
