import nltk
import random
import string
import streamlit as st

f = open('C:/Users/hp/PycharmProjects/Chatbot/data/nlp_data.txt','r', errors='ignore')
data = f.read()

data = data.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_tokens = nltk.sent_tokenize(data)
word_tokens = nltk.word_tokenize(data)

input_greet = ('hi', 'hello', 'how are you', 'hey')
response_greet = ('hi', 'hello', 'yes tell me', 'yeah')


def greet(response):
    for word in response.split():
        if word in input_greet:
            return random.choice(response_greet)


lemment = nltk.WordNetLemmatizer()

def lemmatizing(tokens):
    return [lemment.lemmatize(token) for token in tokens]

remove_punct = dict((ord(punct), None) for punct in string.punctuation)

def Tokenizing(response):
    return lemmatizing(nltk.word_tokenize(response.lower().translate(remove_punct)))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    robo_response = ''
    TfidVec = TfidfVectorizer(tokenizer=Tokenizing, stop_words='english')
    TfidVec = TfidVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(TfidVec[-1], TfidVec)

    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    vals_req = flat[-2]

    if (vals_req == 0):
        robo_response = robo_response + "I can't able to understand you "
        return robo_response
    else:
        robo_response = robo_response + sentence_tokens[idx]
        return robo_response



flag = True
st.write('Hi i am a self learning bot , ask questions to me and if you want to end the convo type bye ')
while (flag == True):
    user_response = st.text_input('Enter you questinon...')
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            st.write(greet(user_response))
        else:
            if (greet(user_response) != None):
                st.write(greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                st.write(response(user_response))
                sentence_tokens.remove(user_response)

    else:
        flag = False
        st.write('Thank you')