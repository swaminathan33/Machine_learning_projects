import pickle
#import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
import numpy as np
#loading vgg16 model
vgg_model = keras.models.load_model("C:/Users/hp/PycharmProjects/ImageCaptionGenerator/models/vgg_model.h5")


#loading our captions
with open('C:/Users/hp/PycharmProjects/ImageCaptionGenerator/captions.txt', 'r') as c:
    next(c)
    doc = c.read()

#loading our model
model = keras.models.load_model("C:/Users/hp/PycharmProjects/ImageCaptionGenerator/models/model.h5")


# load features from pickle
with open('C:/Users/hp/PycharmProjects/ImageCaptionGenerator/models/features.pkl', 'rb') as f:
    features = pickle.load(f)

# mapping the captions with its image id
mapping = {}

for line in tqdm(doc.split('\n')):
    token = line.split(',')
    if len(line) < 2:
        continue
    image_id, captions = token[0], token[1]
    image_id = image_id.split('.')[0]
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(captions)

# clean our data one by one
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)

# getting the next word using the predicted sequence number
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def feature_extracter(image_path, model):
    # extract features from image
    features = []

    image = load_img(image_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features - includes the probability of all labels eg: elephant - 0.2, frog - 0.8
    feature = vgg_model.predict(image, verbose=0)
    # store feature
    features.append(feature)

    return features

def caption_generator(feature, pad_sequences, model, tokenizer, max_length):
    # we give the startseq as input using the features our model predicts the next word
    in_text = 'startseq'
    for i in range(max_length):
        text = tokenizer.texts_to_sequences([in_text])[0]
        text = pad_sequences([text], max_length)
        y_pred = model.predict([feature, text], verbose=0)

        y_pred = np.argmax(y_pred)
        next_word = idx_to_word(y_pred, tokenizer)

        if next_word is None:
            break

        # we do this process again and again until we reach the end sequence ( end_seq )
        in_text += " " + next_word

        if next_word == 'endseq':
            break
    return in_text


