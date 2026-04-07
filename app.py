import streamlit as st
import json
import random
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

with open('intents.json') as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

generator = pipeline("text-generation", model="distilgpt2")

def chatbot_response(user_input):
    # Step 1: Intent Matching
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    if score > 0.3:
        tag = tags[index]
        for intent in data['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    try:
        return wikipedia.summary(user_input, sentences=2)
    except:
        pass

    try:
        response = generator(user_input, max_length=60, num_return_sequences=1)
        return response[0]['generated_text']
    except:
        return "Sorry, I couldn't understand that."
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("Hybrid AI Chatbot")
st.write("Ask anything: College info, general knowledge, or more!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}:** {message}")
