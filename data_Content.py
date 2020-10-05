# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:05:19 2019

@author: Gordana
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading ratings file
# Ignore the timestamp column
#base_dir ='C:\\Users\\Gordana\\ML_Movie_RS\\the-movies-dataset\\'
 
#ratings = pd.read_csv(base_dir+'ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
#users = pd.read_csv(base_dir+ 'users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
#movies = pd.read_csv(base_dir+'movies.csv', sep='\t', encoding='latin-1')


# Import new libraries %matplotlib inline
import wordcloud
from wordcloud import WordCloud, STOPWORDS

# Create a wordcloud of the movie titles
movies['title'] = movies['title'].fillna("").astype('str')
title_corpus = ' '.join(movies['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()
"""