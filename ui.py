import streamlit as st
from model import model, vectorizer

st.title("Spam Classifier")
user_input = st.text_input("Enter a message (English only):")

if user_input:
    prediction = model.predict(vectorizer.transform([user_input]))
    st.write("Spam" if prediction[0] == 'spam' else "Not Spam")