#import libraries 
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib

#collect and preprocess data
data = pd.read_csv('data/movies.csv')
labels = ['genres', 'cast', 'keywords','tagline', 'director']

#clean data
for label in labels:
    data[label] = data[label].fillna('')

titles = data.title

new_data = data['genres'] + data['cast'] + data['keywords'] + data['tagline'] + data['director']
#vecotrorize data 
vectorizer = TfidfVectorizer()

feature_extraction = vectorizer.fit_transform(new_data)

#text = input('Enter a movie name')
st.text_input('Enter a movie name : ', key="name")
text = st.session_state.name

close_matches = difflib.get_close_matches(text, titles)

index = data[data.title == close_matches[0]]['index'].values[0]

similarity = cosine_similarity(feature_extraction)

similarity_score = list(enumerate(similarity[index]))

similarity_similar_movies = sorted(similarity_score,key= lambda x : x[1], reverse=True)

i = 1
for movies in similarity_similar_movies:
    index = movies[0]
    title =  data[data['index'] == index]['title'].values[0]
    if (i<20):
        st.write(title)
        #print(title)
        i+=1