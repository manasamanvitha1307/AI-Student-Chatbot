import os
print("FLASK CWD:", os.getcwd())

from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
import random
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="templates", static_folder="static")

lemmatizer = WordNetLemmatizer()

# Load model + data
model = load_model('chatbot_Application_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))


# Convert sentence → bag of words
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


# Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": labels[r[0]], "probability": str(r[1])} for r in results]


# Generate response
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand."


# Home route (UI)
@app.route("/")
def home():
    return render_template("index.html")


# API route for chatbot
@app.route("/get", methods=["POST"])
def chatbot_reply():
    user_msg = request.json["message"]
    ints = predict_class(user_msg)
    reply = get_response(ints, intents)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)

