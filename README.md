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

    import json
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    data = json.load(file)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        texts.append(" ".join(tokens))
        labels.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Trained Successfully!")

{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi", "Hey", "Good morning"],
      "responses": ["Hello! How can I help you?", "Hi there!"]
    },
    {
      "tag": "fees",
      "patterns": ["What is the fee structure?", "Tell me about fees"],
      "responses": ["You can check the fee structure on the official college website."]
    },
    {
      "tag": "admission",
      "patterns": ["How to apply?", "Admission process"],
      "responses": ["You can apply through the online admission portal."]
    }
  ]
}

