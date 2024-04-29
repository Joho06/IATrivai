from flask import Flask, request, Response ,jsonify
import json
import random
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from spellchecker import SpellChecker
from flask import Flask, request, Response 
from google.cloud import speech
from google.oauth2 import service_account
from pydub import AudioSegment
from google.cloud import texttospeech
import io
import spacy

app = Flask(__name__)
nlp = spacy.load('es_core_news_sm')
# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load preprocessed data and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('IntellichatModel.h5')

# Variable para almacenar el nombre del usuario
user_name = ""

# Functions for chatbot functionality

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    # Ensure that the bag of words has the correct length (57)
    bag = bag[:129]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list if return_list else [{"intent": "no_match", "probability": "0.0"}]

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "Lo siento, no puedo responder esa pregunta en este momento."

def chatbot_response(message, user_id):
    # Si el usuario ya ha proporcionado su nombre, incluirlo en la respuesta
    global user_name
    if user_name:
        message = f"{user_name}, {message}"
    ints = predict_class(message)
    if ints[0]["intent"] != "no_match":
        res = get_response(ints, intents)
        if res is not None:
            return res
        else:
            return "Lo siento, no puedo responder esa pregunta en este momento."
    else:
        return "Lo siento, no entendí tu pregunta."
    
def extract_name(text):
    if not any(char.isupper() for char in text):
        return None
    # Lista de posibles patrones para identificar el nombre
    patterns = ['mi nombre es', 'me llamo', 'soy', 'me dicen']
    
    text_lower = text.lower()
    # Buscar cada patrón en el texto
    for pattern in patterns:
        if pattern in text_lower:
            # Dividir el texto en partes y tomar la última parte como el nombre
            parts = text_lower.split(pattern)
            if len(parts) > 1:
                name = parts[-1].strip()
                return name.capitalize() if name else None
    
    return None


# Routes
@app.route('/chat', methods=['POST'])
def chat_route():
    global user_name  # Declarar user_name como global para modificar la variable global dentro de la función
    data = request.json
    # Establecer user_id en None inicialmente
    user_id = None

    if 'mensaje' in data:
        mensaje_recibido = data['mensaje']
        nombre = extract_name(mensaje_recibido)  # Movido aquí
        # Revisar si 'user_id' está presente en los datos recibidos
        if 'user_id' in data:
            user_id = data['user_id'] 
        # Manejar el caso donde 'user_id' no está presente en data
        else:
            # Puedes asignar un valor predeterminado o devolver un mensaje de error
            user_id = None  
        # Verificar si el mensaje contiene una solicitud para establecer un nombre
        if nombre:
            user_id = nombre
            respuesta = f"¡Bienvenido, {nombre}! ¿En qué puedo ayudarte hoy?"
        else:
            # Verificar si el mensaje coincide con alguno de los patrones del tag "paquetes Colombia"
            respuestas_pais = []
            for intent in intents['intents']:
                if intent['tag'] == data.get('tag'):  # Verificar si el tag coincide con el proporcionado en el JSON
                    for pattern in intent['patterns']:
                        if pattern.lower() in mensaje_recibido.lower():
                            # Agregar todas las respuestas asociadas al tag correspondiente a la lista de respuestas
                            respuestas_pais.extend(intent['responses'])

            # Si se encontraron respuestas asociadas al tag, devolverlas
            if respuestas_pais:
                return jsonify({"respuestas": respuestas_pais})
            else:
                respuesta = chatbot_response(mensaje_recibido, user_id)
    else:
        respuesta = "No se proporcionó un mensaje válido."

    return jsonify({"respuesta": respuesta})


@app.route("/audio", methods=["POST"])
def audioText():
    if 'audio_file' not in request.files:
        return "No se ha proporcionado ningún archivo de audio", 400
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return "No se ha seleccionado ningún archivo", 400
    audio_path = "temp_audio_file.ogg"
    audio_file.save(audio_path)
    convert_ogg_to_wav()
    audio_pathwav = "temp_audio_fileg.wav"
    client_file = "sa_key_demo.json"
    credentials = service_account.Credentials.from_service_account_file(client_file)
    client = speech.SpeechClient(credentials=credentials)
    with open(audio_pathwav, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="es-ES",
        audio_channel_count = 1,
    )
    response = client.recognize(config=config, audio=audio)
    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        print("esta en ok")
        return transcript, 200
    else:
        return "No se encontraron resultados de transcripción", 400

@app.route("/audioToText", methods=["POST"])  
def devolver_audio():
    # Obtener el texto del cuerpo de la solicitud POST
    texto = request.form.get('texto', '')  # Acceder al texto enviado desde PHP
    client_file = "sa_key_demo.json"
    credentials = service_account.Credentials.from_service_account_file(client_file)
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=texto)
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-US",
        name="es-US-Neural2-B",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    print(texto)
    return Response(response.audio_content, mimetype='audio/mpeg')

def convert_ogg_to_wav():
    song = AudioSegment.from_ogg("temp_audio_file.ogg").set_sample_width(2)
    song.export("temp_audio_fileg.wav", format="wav")
if __name__ == "__main__":
    '''mensaje_usuario = "ayudame con informacion"
    respuesta = mensaje_Llega(mensaje_usuario)
    print(respuesta)'''
    app.run(debug=True)
    #devolver_audio("texto")