import streamlit as st
import pickle
import nltk
import json
import random
from nltk.stem import WordNetLemmatizer

# Download required data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

st.title("🎓 College ML Chatbot")

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    with open("intents.json") as file:
        intents = json.load(file)

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

user_input = st.text_input("Ask your query:")

if user_input:
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    response = get_response(prediction)
    st.write("Bot:", response)