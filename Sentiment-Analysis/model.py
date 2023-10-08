import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
import pickle
import streamlit


data = pd.read_csv('C:/Users/hp/PycharmProjects/Sentiment-Analysis/Sentiment.csv')
data = data[['text','sentiment']]
data = data[data['sentiment'] != 'Neutral']

#data cleaning
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt','')

#Vectorizing
tokenizer = Tokenizer(2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

y = pd.get_dummies(data['sentiment'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#training
model = keras.Sequential()

model.add(Embedding(2000, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
""" 

print('Fitting....')

model.fit(X_train, y_train, epochs=7)                     ----------> remove this comment when you are running

print(model.evaluate(X_test, y_test))

print('----------Completed-----------')

print('Copying model............')

with open('model.pkl','wb') as f:
    pickle.dump(model, f)

"""

with open('C:/Users/hp/PycharmProjects/Sentiment-Analysis/model.pkl', 'rb') as f:
    m = pickle.load(f)


def predictor():
    text = streamlit.text_input('Enter the sentence ')
    if text:
        tokenizer = Tokenizer(2000, split=' ')
        tokenizer.fit_on_texts(text)
        sentence = tokenizer.texts_to_sequences(text)
        sentence = pad_sequences(sentence)
        pred = m.predict(sentence)
        if np.argmax(pred) == 0:
            streamlit.write('Negative Sentence')
        else:
            streamlit.write('Positive Sentence')


predictor()

