import pickle

import streamlit as st
from PIL import Image
import numpy as np


def predictor(img_file):
    pil_img_file = Image.open(img_file)
    img_file = pil_img_file.convert('RGB')
    img_file = img_file.resize((128,128))
    img_file = np.array(img_file) / 255
    return np.array(img_file).reshape([1,128,128,3]), pil_img_file


model = pickle.load(open('./model.pkl','rb'))

upload_file = st.file_uploader('Choose a file')
if upload_file:
    img, pil_img_file = predictor(upload_file)
    prediction = np.argmax(model.predict(img))
    if prediction == 0:
        caption = 'The person not wearing a mask'
    else:
        caption = 'The person wearing mask '
    st.image(pil_img_file, caption=caption)
