# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:51:34 2019

@author: Gordana
"""

#!/usr/bin/env python

from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import scipy.spatial.distance as ssd
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from surprise.model_selection import  KFold
from sklearn.metrics import mean_squared_error
import time
from math import sqrt
from itertools import permutations 
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import GridSearchCV, KFold, train_test_split
from surprise.model_selection.validation import cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVDpp


def cosine_distances(X, Y):
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    return 1. - ssd.cdist(X, Y, 'cosine')



class UserSimilarity(object):
    def __init__(self, modelRatings,  modelMovies):
        self.modelRatings = modelRatings
        self.modelMovies = modelMovies



    def SvDNew (self, parFactor, parLearningRate, parRegular):
        reader =Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.modelRatings[['userId', 'movieId', 'rating']], reader)
        start = time.time()
        kf = KFold(n_splits=5)
        algo = SVD(n_factors =parFactor, lr_all = parLearningRate,reg_all = parRegular, verbose=True)
        rmses =[]
    
        for  in 
            # train and test algorithm.
            algo.fit(trainset)
            predictions = algo.test(testset)
            # Compute and print Root Mean Squared Error
            #accuracy.rmse(predictions, verbose=True)
            rmses.append(accuracy.rmse(predictions, verbose=True))
            print("Runtime %0.2f" % (time.time() - start))
            
        print(rmses)
        rmse_avg = round(sum(rmses) / len(rmses),5)
        print('The mean RMSE of the full rating set is: {}'.format(rmse_avg))
        
        return rmse_avg 
        

    def analysis(self):
        lsLearning =[0.001, 0.005, 0.004, 0.003,0.002]
        #lsFactor = [100, 14, 20 ,50, 80]
        lsRegular =[0.02, 0.01, 0.04, 0.03, 0.02 ]
        rmse = []
        for l in lsLearning:
            for r in lsRegular:
                rs = self.SvDNew( 100, l, r)
                rmse.append((l, r, rs))
        return lsLearning, lsRegular, rmse
    
    def svdM(self, model,  k):
       
        #R_df.head()
        R = model.as_matrix()
        user_ratings_mean = np.mean(R, axis = 1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(R_demeaned, k = k)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = model.columns)
        
        return preds_df 

    

    def fit(self):
        # define a cross-validation iterator
        start = time.time()
        kf = KFold(n_splits=5)
        lsRMSE={}
        k = 50
        for i in range(5):
            trainset, testset =train_test_split(self.modelRatings, test_size=0.2)
            R_df = trainset.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
            test_df= testset.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
            predictions_df = self.svdM(R_df, k)
            val_rmse= self.rmseM(predictions_df, test_df)
            print ('User-based CF MSE: ' + str(val_rmse))
            print("Runtime %0.2f" % (time.time() - start))
            lsRMSE[k] = val_rmse
            k +=2
        return lsRMSE   
    
    def recommend_movies(self, predictions_df, userID, num_recommendations=5):
        user_row_number = userID # UserID starts at 1, not 0
        sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
        # Get the user's data and merge in the movie information.
        user_data = self.modelRatings[self.modelRatings.userId == (userID)]
        user_full = (user_data.merge(self.modelMovies, how = 'left', left_on = 'movieId', right_on = 'id').
                     sort_values(['rating'], ascending=False)
                 )
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations = (self.modelMovies[~self.modelMovies['id'].isin(user_full['movieId'])].
                                     merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',left_on = 'id',right_on = 'movieId').
                                     rename(columns = {user_row_number: 'Predictions'}).
                                     sort_values('Predictions', ascending = False).
                                     iloc[:num_recommendations, :-1])

        return user_full, recommendations


        
def start():
    movies = load_dataset()
    

if __name__ == '__main__':
    #start()
     movies = load_dataset()