# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:34:55 2019

@author: Gordana
"""

#!/usr/bin/env python
from data import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re 
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk #nltk.download('punkt')
from nltk.tokenize import word_tokenize


from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import scipy.spatial.distance as ssd
from scipy.sparse.linalg import svds
from surprise.model_selection import  KFold
from sklearn.metrics import mean_squared_error
import time
from math import sqrt
from itertools import permutations 
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import GridSearchCV, KFold, train_test_split
from surprise.model_selection.validation import cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVDpp

def start():
    movies = load_dataset()
    
  
# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title):
    idx = indices[title]# vrati normalnlno
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
     
def doc2VecImplementation (mov_model):
    
    mov_model['overview'] = mov_model['overview'].map(lambda x: x.lower())
    all_docs = list(zip(mov_model['overview'], mov_model['id']))
    tagged_docs = [TaggedDocument(words = word_tokenize(doc.lower()), tags = [str(pos)]) for doc, pos in all_docs]
    #tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    #mov_model.at[0,'clean_text'] #print tagged_docs[0]
    model = Doc2Vec(vector_size = 150,
                alpha = 0.025, 
                min_alpha = 0.00025,
                min_count = 2,
                dm = 1)
    # Builds the vocabulary from all of the documents
    model.build_vocab(tagged_docs)
    max_epochs = 20
    for epoch in range(max_epochs):    
        if epoch % 5 == 0:
            print(f'Processing epoch number: {epoch}')
        
    model.train(tagged_docs,
                total_examples=model.corpus_count,
                epochs=model.iter)
    
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    model.save("d2v.model")
    #model.save("../models/d2v.model")
    print("Model Saved")
    return model 

def predicted_top_n(Id, n, df, movies_df, algo):
    '''
    This function returns n movies, sorted by predicted user rating, from a random sample of movies. 
    '''
    df = df[['movieId','userId','rating']]

    movie_choices = df['movieId'].unique()
    
    # Take out movies the user has already watched
    temp_df = df[df['userId'] == Id]
    watched_movs = temp_df['movieId'].unique()
    unwatched = np.setdiff1d(movie_choices,watched_movs)
    movies = unwatched
        
    # Build the dataframe that we'll return
    predicted_df = pd.DataFrame()
    predicted_df['movieId'] = movies
    predicted_df['userId'] = Id
    predicted_df['est'] = predicted_df['movieId'].apply(lambda x: round(algo.predict(Id,x).est,2))
    predicted_df = predicted_df.sort_values(by='est', ascending=False)
    predicted_df = predicted_df.head(n)
    predicted_df = pd.merge(predicted_df,movies_df ,left_on='movieId', right_on ='id')
    return predicted_df[['userId','title','est']]

    

def check_system(Id,limit,df,movies,algo):
    # Isolates necessary columns from the dataframe
    df = df[['movieId','userId','rating']]
    
    # Takes a subsample of the user's ratings
    user_df = df[df['userId'] == Id]
    if user_df.shape[0] >= df['userId'].value_counts().mean():
        user_df = user_df.sample(frac=.10)
    else:
        user_df = user_df.sample(frac=.50)

    # Builds the dataframe to be returned     
    user_df['est'] = user_df['movieId'].apply(lambda x: round(algo.predict(Id,x).est,2))
    user_df['error'] = user_df['est']-user_df['rating']
    user_df['avg_error'] = user_df['error'].mean()
    
    # Returns a dataframe dependent on what the limit is set to
    if limit == None:
        user_df = pd.merge(user_df,movies,left_on='movieId', right_on='id')
        return user_df[['userId','movieId','title','rating','est','error','avg_error']]
    else:
        if limit >= user_df.shape[0]:
            user_df = pd.merge(user_df,movies,left_on='movieId', right_on='id')
            return user_df[['userId','movieId','title','rating','est','error','avg_error']]
        else:
            user_df = user_df.head(limit)
            user_df = pd.merge(user_df,movies,left_on='movieId', right_on='id')
            return user_df[['userId','movieId','title','rating','est','error','avg_error']]

def svd_my (model ):
    
    reader = Reader(rating_scale=(1, 5))
    start = time.time()
    algo = SVD(verbose=True)
    data = Dataset.load_from_df(model[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=.25)
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
    print("Runtime %0.2f" % (time.time() - start))
    return predictions, algo 
    # Compute and print Root Mean Squared Error
    #accuracy.rmse(predictions, verbose=True)
    

def  tuning (data):
    trainset, testset = train_test_split(data, test_size=.20)
    reader = Reader(rating_scale=(1, 5))
    lsLearning  =[0.0, 0.5, 0.2, 0.08, 0.005, 0.002, 0.001, 0.001]
    #data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
    #datas = data.build_full_trainset()
    #test =  Dataset.load_from_df(test[['userId', 'movieId', 'rating']], reader)
    #tests= test.build_full_trainset()
    rmseLearning =[]
    rmseRegular =[]
    for item in lsLearning:
        algo = SVD(verbose=True, lr_all= item, n_epochs = 5)
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmseLearning.append((item, accuracy.rmse(predictions, verbose=True)))
    lsRegular =[0.7, 0.5, 0.4, 0.6, 0.4, 0.03, 0.01, 0.02]
    for item in lsRegular:
        algo = SVD(verbose=True, reg_all= item,  n_epochs = 5)
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmseRegular.append((item, accuracy.rmse(predictions, verbose=True)))
    return rmseLearning , rmseRegular


if __name__ == '__main__':
    #start()
     movies = load_dataset()
     mov_title=[]
     mov_genres =[]
     reader = Reader(rating_scale=(1, 5))
     #data = Dataset.load_from_df(model[['userId', 'movieId', 'rating']], reader)
     #model_coll = UserSimilarity( movies['item_ids'], movies['data'])
     #preds_df =
     #plot = model_coll.analysis()
     #p, alg = svd_my(movies['itemUser'])
     
     #recom=  recommend_movies(UserSimilarity( movies['item_ids'], movies['data']), matrix_pred, 837, 10)
     
     
     #for (id, title) in movies['item_ids'].keys():
         #mov_title.append(title)
         
     #already_rated, predictions = model_coll.recommend_movies(preds_df, 837, 10)
     """
     tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

     tfidf_matrix = tf.fit_transform(list(movies['data']['bag_of_words']))
     tfidf_matrix.shape   
     cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
     cosine_sim[:4, :4]
     # Build a 1-dimensional array with movie titles
     
     titles = movies['data']['title']
     indices = pd.Series(movies['data']['title'], index=movies['data']['index_mod'])
     genre_recommendations('Inception').head(20)
     """
     model_doc2 = doc2VecImplementation(movies['data'])
     #userFeatures = neMov.groupby('userId')['rating'].mean()
     
     """
     vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
print(vector)
     ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])
    
   # Create a wordcloud of the movie titles
       title_corpus = ' '.join(mov_title)
     title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

    # Plot the wordcloud
     plt.figure(figsize=(16,8))
     plt.imshow(title_wordcloud)
     plt.axis('off')
     plt.show()
"""
def CheckDoc2Vec ( model_doc2, movies, user_rating):
    # Takes a subsample of the user's ratings
    userFeatures = user_rating.groupby('userId')['rating'].mean()
    userId = userFeatures.index
    data ={'userId' :[] , 'movieId':[], 'rating':[], 'est':[], 'error':[]}
    for userRat in userId[50:160]:
        lsId =movies['id'].tolist()
        #print(lsId)
        movieId = int(user_rating[(user_rating['userId']==userRat) & (user_rating['movieId'].isin(lsId))].head(1)['movieId'])
        data['userId'].append(userRat)
        data['movieId'].append(movieId)
        pdOver= movies[movies['id']==movieId]['overview']
        inferred_vector = model_doc2.infer_vector(pdOver.to_string().split()[1:])
        sims = model_doc2.docvecs.most_similar([inferred_vector], topn=10)
        iSum = sum(list(map(lambda x: x[1], sims)))
        lsMovieRat = [ float(movies[movies['id']==int(i[0])]['weighted_rating']) for i in sims ]
        cros_pro= sum([sims[i][1]*lsMovieRat[i] for i in range(len(sims)) ])
        rui= userFeatures.iloc[userRat] + (cros_pro/iSum)
        data['rating'].append(float(user_rating[(user_rating['userId']==userRat) & (user_rating['movieId']==movieId)]['rating']))
        data['est'].append(rui)
        data['error'].append(rui-data['rating'][-1])
    return pd.DataFrame.from_dict(data)

                              
                              
            
        
    # Builds the dataframe to be returned     
    
     