import streamlit as st
from notebook import pad_sequences, model, tokenizer, max_length, caption_generator, feature_extracter

image = st.file_uploader('Choose a photo ')
if image:
    st.write('Extracting ⛏️✂️⚒️ ')
    features = feature_extracter(image, model)
    st.write('Predicting 🤔🔎')
    y_pred = caption_generator(features, pad_sequences, model, tokenizer, max_length)
    st.write('captions 👇👇👇')
    st.write('--------------------------------')
    st.write(y_pred)
    st.write('--------------------------------')
    st.image(image)