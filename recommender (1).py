import logging
import numpy as np
import pandas as pd
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator


class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self,number_features=15,reg_parameter=.05,maxIter = 20):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        # ...
        self.spark = SparkSession.builder.master("local[4]").getOrCreate()
        self.K = number_features
        self.reg_parameter = reg_parameter
        self.train_mean = None
        self.col_filter = None
        self.users_fit = None
        self.movies_fit = None
        self.user_means = {}
        self.movie_means = {}
        self.maxIter = maxIter
        self.gbtmodel = None

    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")
        train = self.spark.createDataFrame(ratings)
        train = train.drop('timestamp')
        #Calculate The training mean
        self.train_mean = train.agg(avg(col("rating"))).collect()[0]['avg(rating)']

        #train the ALS model for collaborative filtering
        als_model = ALS(itemCol = 'movie',userCol = 'user',
                        ratingCol = 'rating',nonnegative=True,
                        regParam=self.reg_parameter,rank=self.K,
                        maxIter = self.maxIter)
        als_model = als_model.fit(train)
        self.col_filter = als_model
        self.logger.debug("finishing fit")

        #determining which users and movies have been used in training.
        self.users_fit = {u.user for u in train.select('user').distinct().collect()}
        self.movies_fit = {m.movie for m in train.select('movie').distinct().collect()}

        #determining the average rating in the training set for each user and movie in the set
        user_avgs = train.groupby('user').agg(avg('rating')).toPandas().set_index('user')
        movie_avgs = train.groupby('movie').agg(avg('rating')).toPandas().set_index('movie')
        for user in self.users_fit:
            self.user_means[user] = user_avgs.loc[user][0]
        for movie in self.movies_fit:
            self.movie_means[movie] = movie_avgs.loc[movie][0]

        #determining the variation from the mean for each movie/user in the training set (bu_i and bv_j)
        self.bui = {mu_u - self.train_mean for user, mu_u in self.user_means.items()}
        self.bvj = {mu_m - self.train_mean for movie, mu_m in self.user_means.items()}

        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))



        #transforming data using collaborative filtering
        requests_sp = self.spark.createDataFrame(requests)
        req_pred = self.col_filter.transform(requests_sp)
        req_pred = req_pred.toPandas()
        req_pred = req_pred.rename(index=str,columns={'prediction':"rating"})

        #preparing data for GB model predictions
        def coldstart_helper(row):
            if (row.user not in self.users_fit) and (row.movie not in self.movies_fit):
                return self.train_mean
            elif row.user not in self.users_fit:
                return self.movie_means[row.movie]
            else:
                return self.user_means[row.user]

        #applies cold start helper to values that our collaborative filtering could not predict
        req_pred['rating'] = req_pred.apply(lambda x: coldstart_helper(x) if (pd.isnull(x.rating)) else x['rating'], axis = 1)

        self.logger.debug("finishing predict")
        return(req_pred)

    def _evaluate(self,test_data,metric = "rmse"):
        test_data = self.spark.createDataFrame(test_data)
        predictions = self.col_filter.transform(test_data)
        predictions = predictions.na.fill(self.train_mean)
        eva = RegressionEvaluator(metricName=metric, labelCol="rating",
                                  predictionCol="prediction")
        score = eva.evaluate(predictions)
        return score

if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
