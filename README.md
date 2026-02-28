import streamlit as st
import pickle
import nltk
import json
import random

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

st.title("🎓 College ML Chatbot")

user_input = st.text_input("Ask your query:")

if user_input:
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    response = get_response(prediction)
    st.write("Bot:", response)
