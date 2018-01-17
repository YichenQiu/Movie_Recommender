from uszipcode import ZipcodeSearchEngine
import pandas as pd
import numpy as np


def find_user_vector(id, type_):
    table = merge_data()
    if type_ = 'MovieID'

    return table[table['UserID'] == 'userid']


def merge_data():
    user_df, movie_df, train_df = clean_data()
    #train_df = pd.read_csv(trainpath)
    #("/Users/willhd/galvanize/dsi-recommender-case-study/data/training.csv")
    train_df = train_df.drop('timestamp', axis=1)
    test = train_df.merge(movie_df, left_on='movie', right_on='MovieID').merge(
        user_df, left_on='user', right_on='UserID')
    merged_df = test.drop(columns=['user', 'Gender', 'Zip', 'movie', 'regions'])
    return merged_df


def clean_data():
    movie = clean_movie()
    user = clean_user()
    train = clean_train()
    return user, movie


def clean_train(path):
    train_df = pd.read_csv(path)
    #("/Users/willhd/galvanize/dsi-recommender-case-study/data/training.csv")
    train_df = train_df.drop('timestamp', axis=1)


def clean_movie():
    movie1 = pd.read_table("data/movies.dat", delimiter="::", names=["MovieID", "Title", "Genres"])
    movie1["Movie"], movie1['Year'] = movie1.Title.apply(
        lambda x: x[:-6]), movie1.Title.apply(lambda x: x[-6:])
    movie1["Year"] = movie1["Year"].str.strip("(")
    movie1["Year"] = movie1["Year"].str.strip(")")
    movie1["Genres"] = movie1.Genres.apply(lambda x: x.split("|"))
    movie2 = movie1.join(pd.get_dummies(movie1.Genres.apply(pd.Series).stack()).sum(level=0))
    movie3 = movie2.drop(columns=['Title', "Genres"])
    movie3["Year"] = movie3['Year'].astype(int)
    movie4 = movie3.drop(columns=["MovieID"])
    return movie4


def clean_user(userpath):
    users1 = pd.read_table(userpath, delimiter="::", names=["UserID", "Gender", "Age",
                                                            "Occupation", "Zip"])
    users1['regions'] = users1.Zip.apply(lambda x: convert_zip_to_region(x))
    users1 = users1.join(pd.get_dummies(users1.regions))
    users1 = users1.join(pd.get_dummies(users1.Gender))
    # will keep this incase we decide to use logistic regression
    # users1.join(pd.get_dummies(users1.Age))
    # users1['13'] = users1['Age'] <= 13
    # users1['26'] = (users1['Age'] > 14) & (users1['Age'] <= 26)
    # users1['39'] = (users1['Age'] > 26) & (users1['Age'] <= 39)
    # users1['56'] = (users1['Age'] > 39) & (users1['Age'] <= 56)
    users1 = users1.join(pd.get_dummies(users1.Occupation))
    #users1 = users1.drop(columns=['Zip', 'Occupation', 'Gender', 'regions'])
    return users1


def convert_zip_to_region(zipcode):
    state_region_dct = {'AK': 'West',
                        'AL': 'South',
                        'AR': 'South',
                        'AZ': 'West',
                        'CA': 'West',
                        'CO': 'West',
                        'CT': 'Northeast',
                        'DC': 'South',
                        'DE': 'South',
                        'FL': 'South',
                        'GA': 'South',
                        'HI': 'West',
                        'IA': 'Midwest',
                        'ID': 'West',
                        'IL': 'Midwest',
                        'IN': 'Midwest',
                        'KS': 'Midwest',
                        'KY': 'South',
                        'LA': 'South',
                        'MA': 'Northeast',
                        'MD': 'South',
                        'ME': 'Northeast',
                        'MI': 'Midwest',
                        'MN': 'Midwest',
                        'MO': 'Midwest',
                        'MS': 'South',
                        'MT': 'West',
                        'NC': 'South',
                        'ND': 'Midwest',
                        'NE': 'Midwest',
                        'NH': 'Northeast',
                        'NJ': 'Northeast',
                        'NM': 'West',
                        'NV': 'West',
                        'NY': 'Northeast',
                        'None': 'West',
                        'OH': 'Midwest',
                        'OK': 'South',
                        'OR': 'West',
                        'PA': 'Northeast',
                        'PR': 'Other',
                        'RI': 'Northeast',
                        'SC': 'South',
                        'SD': 'Midwest',
                        'TN': 'South',
                        'TX': 'South',
                        'UT': 'West',
                        'VA': 'South',
                        'VT': 'Northeast',
                        'WA': 'West',
                        'WI': 'Midwest',
                        'WV': 'South',
                        'WY': 'West'}

    search = ZipcodeSearchEngine()
    zipcode = search.by_zipcode(zipcode)
    state = zipcode.State
    if state is not None:
        region = state_region_dct[state]
        return region
