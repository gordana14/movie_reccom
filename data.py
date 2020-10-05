from os.path import dirname
from os.path import join
import numpy as np
import pandas as pd
import re 
from sklearn.model_selection import train_test_split
import random 
# ast module for abstract syntax grammar
from ast import literal_eval
import ast

class Data(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self



def load_dataset(*args):
    if args:
        if len(args) == 2:
            rating_file = args[0]
            movie_file = args[1]
        else:
            print ('Please provide both rating and movie file')
    else:
        rating_file = 'ratings.csv'
        movie_file = 'movies_metadata.csv'
    
    #base_dir = join(dirname(__file__), 'data/')
    base_dir ='C:\\Users\\Gordana\\ML_Movie_RS\\the-movies-dataset\\'
    rating_file = 'ratings.csv'
    movie_file = 'movies_metadata.csv'

    #Read the titles
    df_movies = pd.read_csv(base_dir + movie_file)
    df_movies['release_data']= pd.to_datetime(df_movies['release_date'])
    is_greater_2009 = df_movies['release_date']>='2009-01-01'
    df_movies2009 = df_movies
    df_movies2009['tagline'] = df_movies['tagline'].astype('str') 
    df_movies2009['overview'] = df_movies['overview'].astype('str') 
    mov_genres= []
    count_row =[]
    mov_tagline =[]
    counts=0
    for val , tag , overview in zip(df_movies2009['genres'], df_movies2009['tagline'], df_movies2009['overview']):
         counts+=1
         #8351 empty
         if tag=='nan':
             lsSplitOverview= overview.split(' ')
             lsImpKey= [s for s in lsSplitOverview if len(s)>5]
             newtag = ' '.join(lsImpKey[3:int(len(lsImpKey)/2) + 3])
             mov_tagline.append(newtag)
         else:
             mov_tagline.append(tag)
             
         if len(val)==2:
             mov_genres.append(' ')
             count_row.append(counts)
         else:
             regex =re.compile("\{'id': \d{1,8}, \'name\': \'")
             lsSplit = list(regex.split(val))[1:]
             new_item = ''
             for item in lsSplit:
                 if ']' in item:
                    new_item+=item[:-3]
                 else:
                    new_item+=item[:-4]+'|'
             mov_genres.append(new_item)
             count_row.append(counts)
             
    vote_counts = df_movies2009[df_movies2009['vote_count'].notnull()]['vote_count'].astype('int')
    m = vote_counts.quantile(0.75)
    vote_averages = df_movies2009[df_movies2009['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    
    def weighted_rating(x):
        v = x['vote_count']+1 # added +1 - Dan
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    weight_rat = df_movies2009.apply(weighted_rating, axis=1)
    dfObj = pd.DataFrame(df_movies2009, columns=['title'])
    duplicateRowsDF = dfObj[dfObj.duplicated()].index
    df_movies2009['genres'] = df_movies2009['genres'].fillna('[]').apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # Production companies
    df_movies2009['production_countries'] = df_movies2009['production_countries'].fillna("[]").apply(ast.literal_eval)
    df_movies2009['production_countries'] = df_movies2009['production_countries'].apply( lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    
    # Belongs to collection
    df_movies2009['belongs_to_collection'] = df_movies2009['belongs_to_collection'].fillna("[]").apply(ast.literal_eval).apply( lambda x: x['name'] if isinstance(x, dict) else np.nan)
   
    df_movies2009.insert(loc=len(df_movies2009.columns), column='genres_mod', value=mov_genres) 
    df_movies2009.insert(loc=len(df_movies2009.columns), column='index_mod', value=count_row) 
    df_movies2009.insert(loc=len(df_movies2009.columns), column='tagline_mod', value=mov_tagline)     
    df_movies2009.insert(loc= len(df_movies2009.columns), column = 'weighted_rating', value = weight_rat)
    df_movies2009.drop(duplicateRowsDF, inplace=True)
    
    df_movies2009['weighted_rating']=df_movies2009['weighted_rating'].apply(lambda x: x/2.0 if x >2.5 else x )
    # Read ratings
    df_ratings = pd.read_csv(base_dir + rating_file)
    #189  = is_greater_2009.min()
    df_ratings['rating'] = df_ratings['rating'].apply( lambda x: x+0.7 if x<1.0 else x )

    id_greater_2009 =  df_ratings['movieId'].isin(df_movies2009.id.tolist())
    df_ratings2009 = df_ratings[id_greater_2009]
    #train_data, test_data = train_test_split(df_ratings2009, test_size=0.2)
    #data_m.describe()
    df_movies2009['bag_of_words'] = df_movies2009['tagline_mod'].map(str) +" " + df_movies2009['genres_mod'].map(str) +" " +df_movies2009['production_countries'].map(str)
    return Data(data=df_movies2009, itemUser=df_ratings,user_ids=None)

