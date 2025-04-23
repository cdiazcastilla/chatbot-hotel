from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import nltk
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
import random

# Descargas necesarias
nltk.download('punkt')
nltk.download('wordnet')

# Inicializaciones
app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())

# Crear lista de palabras y clases
words = []
classes = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
    classes.append(intent['tag'])

# Lemmatizar y limpiar
words = [lemmatizer.lemmatize(w.lower()) for w in words]
words = sorted(list(set(words)))[:50]  # Limitar a 50 palabras
classes = sorted(list(set(classes)))

# Funciones de procesamiento
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model, threshold=0.6):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    index = np.argmax(res)
    if res[index] > threshold:
        return classes[index]
    else:
        return "noanswer"  # Asegúrate de que exista este tag en intents.json

def get_response(intent, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])

# Rutas de la API
@app.route("/")
def home():
    return "¡Hola! Estoy aquí para ayudarte con el hotel."

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    if not user_input:
        return jsonify({"response": "No se recibió ningún mensaje."})
    intent = predict_class(user_input, model)
    response = get_response(intent, intents)
    if response:
        return jsonify({"response": response})
    else:
        return jsonify({"response": "Lo siento, no entendí eso. ¿Puedes repetirlo?"})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)


