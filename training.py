import nltk
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Descargar los recursos necesarios de nltk
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Cargar el archivo intents.json
intents = json.loads(open('intents.json').read())

# Inicialización de listas para almacenar datos de entrenamiento
words = []
classes = []
documents = []
ignore_words = ['?','!','.',',']

# Recopilar palabras y clases de los patrones
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenización de cada palabra en el patrón
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((pattern, intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lematizar y quitar palabras duplicadas
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Ordenar las clases
classes = sorted(list(set(classes)))

# Guardar las palabras y clases en archivos pickle
import pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear los datos de entrada para el modelo
training_sentences = []
training_labels = []

# Crear los datos de entrenamiento
for doc in documents:
    sentence = doc[0]
    word_list = nltk.word_tokenize(sentence)
    word_list = [lemmatizer.lemmatize(w.lower()) for w in word_list]
    
    # Crear la bolsa de palabras
    bag = [1 if w in word_list else 0 for w in words]
    training_sentences.append(bag)
    training_labels.append(classes.index(doc[1]))

# Convertir a arrays de numpy
training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_dim=len(training_sentences[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

# Entrenar el modelo
model.fit(training_sentences, training_labels, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save('chatbot_model.h5')

print("Modelo entrenado y guardado como 'chatbot_model.h5'")
