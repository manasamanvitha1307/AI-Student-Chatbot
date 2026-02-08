import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
import json
import random
from keras.models import load_model

model = load_model('chatbot_Application_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

def bag_of_words(s, words):
    bag = [0] * len(words)
    sent_words = nltk.word_tokenize(s)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]

    for sent in sent_words:
        for i, w in enumerate(words):
            if w == sent:
                bag[i] = 1
    return np.array(bag)

def predict_label(s):
    pred = bag_of_words(s, words)
    response = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(response) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": labels[r[0]], "probability": str(r[1])} for r in results]

def Response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def chatbot_response(msg):
    ints = predict_label(msg)
    return Response(ints, intents)

def chat():
    print("Start chat with ChatBot of QIS")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        print("\nBOT:", chatbot_response(inp), "\n")

chat()