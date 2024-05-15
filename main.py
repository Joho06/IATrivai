from flask import Flask, request, Response ,jsonify
import json
import random
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from google.cloud import speech
from google.oauth2 import service_account
from pydub import AudioSegment
from google.cloud import texttospeech
import io
import spacy
from googletrans import Translator


app = Flask(__name__)
nlp = spacy.load('es_core_news_sm')
# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
confirming = False
user_name = ''

    
# Load intents data from JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load preprocessed data and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('IntellichatModel.h5')

# Inicializar traductor
translator = Translator()

# Functions for chatbot functionality
# Definir una función para manejar la confirmación de viaje
# def handle_confirmation():
#     # Inicializar un diccionario para almacenar la información del viaje
#     trip_info = {}

#     # Mostrar el mensaje de confirmación
#     confirm_message = "Para poder ayudarte necesitamos saber la siguiente información:"

#     return trip_info, confirm_message


# # Obtener la respuesta del usuario al patrón de confirmación de viaje
# def get_confirmation_response(confirmation_questions):
#     # Solicitar y guardar la información del usuario
#     trip_info, confirm_message = handle_confirmation()

#     return confirm_message, trip_info  # Devolver el mensaje de las respuestas del usuario con formato de salto de línea

# confirmation_questions = [
#     "Nombre:",
#     "Correo electrónico:",
#     "Destino:",
#     "Fecha tentativa:",
#     "Cuantas personas viajan:",
#     "Edades:",
#     "Salida de Quito o Guayaquil:"
# ]



# def guardar_en_json(data, filename):
#     with open(filename, 'w') as json_file:
#         json.dump(data, json_file, indent=4)


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
    # Ensure that the bag of words has the correct length (142)
    bag = bag[:100000]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list if return_list else [{"intent": "no_match", "probability": "0.0"}]

def get_response(ints, intents):
    tag = ints[0]['intent']
    list_of_intents = intents.get('intents', [])  # Usa get() para manejar el caso en que 'intents' no esté definido
    for i in list_of_intents:
        if i['tag'] == tag:
            responses = i.get('responses', [])  # Usa get() para manejar el caso en que 'responses' no esté definido
            if responses:
                result = random.choice(responses)
                return result
    return "Lo siento, no puedo responder esa pregunta en este momento."


def chatbot_response(message, user_id):
    global user_name
    global confirmation_info
    # Verifica si el user_name ya está establecido y usa ese valor para personalizar el mensaje
    if user_name:
        message = f"{user_name}, {message}"
    
    # Agrega una verificación para asegurarte de que el mensaje sea suficientemente largo y compuesto de caracteres alfanuméricos
    if len(message) < 3 or not any(char.isalnum() for char in message):
        return "Lo siento, no entendí tu pregunta."
    
    # Verifica si el mensaje contiene nombres de países mencionados
    mentioned_countries = ["Panamá", "Colombia", "Argentina", "Brasil", "Caribe", "Bahamas", "Cuba", "Peru"]  
    countries_mentioned = [country for country in mentioned_countries if country.lower() in message.lower()]
    
    if countries_mentioned:
        # Si hay países mencionados, procesa el mensaje en consecuencia
        respuestas_pais = []
        for country in countries_mentioned:
            # Obtener las respuestas asociadas con el país mencionado
            intents_for_country = [intent for intent in intents.get('intents', []) if country.lower() in intent.get('patterns', [''])[0].lower()]
            for intent in intents_for_country:
                for response in intent.get('responses', []):
                    if response.strip() and message.strip() and response.split()[0].lower() in message.split()[0].lower():
                        respuestas_pais.append(response)
                respuestas_pais.extend(intent.get('responses', []))
                
        respuesta = ' '.join(respuestas_pais) if respuestas_pais else "Lo siento, no tengo información sobre esos países."
    else:
        # Si no hay países mencionados, procesa el mensaje como de costumbre
        ints = predict_class(message)
        respuesta = get_response(ints, intents)
        if respuesta is None:
            respuesta = "Lo siento, no puedo responder esa pregunta en este momento."
        elif ints[0]["intent"] == "confirmacion":
            # Si la intención es confirmación, llama a la función para manejar la confirmación
            # y luego devuelve un mensaje apropiado
            confirm_message = "Para poder ayudarte necesitamos saber la siguiente información:"
            resultado = "¡Gracias! La información ha sido guardada."
            return confirm_message, confirmation_questions, resultado
            
    return respuesta

def extract_name(text):
    if not any(char.isupper() for char in text):
        return None
    # Lista de posibles patrones para identificar el nombre
    patterns = ['mi nombre es', 'me llamo', 'me dicen']
    
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
        nombre = extract_name(mensaje_recibido)  
        if nombre:
            user_id = nombre
           
            respuesta = f"¡Bienvenido, {nombre}! ¿En qué puedo ayudarte hoy?"
        else:
            respuestas_pais = []
            for intent in intents.get('intents', []):
                if intent['tag'] == data.get('tag', ''):
                    for pattern in intent['patterns']:
                        if pattern.lower() in mensaje_recibido.lower():
                            respuestas_pais.extend(intent.get('responses', []))
            if respuestas_pais:
                return jsonify({"respuestas": respuestas_pais})
            else:
                
                respuesta = chatbot_response(mensaje_recibido, user_id)
                if isinstance(respuesta, tuple):
                    # Si la respuesta es una tupla, significa que es el mensaje de confirmación y la información del viaje
                    confirm_message, confirmation_questions = respuesta
                    return jsonify({"confirmacion": confirm_message, "preguntas_confirmacion": confirmation_questions})
                else:
                    # Si no es una tupla, es la respuesta del chatbot
                    return jsonify({"respuesta": respuesta})
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
    #client_file = "sa_key_demo.json"
    #credentials = service_account.Credentials.from_service_account_file(client_file)
    #client = texttospeech.TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=texto)
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-US",
        name="es-US-Neural2-B",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    #response = client.synthesize_speech(
     #   input=synthesis_input, voice=voice, audio_config=audio_config
    #)

    print(texto)
    #return Response(response.audio_content, mimetype='audio/mpeg')

def convert_ogg_to_wav():
    song = AudioSegment.from_ogg("temp_audio_file.ogg").set_sample_width(2)
    song.export("temp_audio_fileg.wav", format="wav")
if __name__ == "__main__":
    '''mensaje_usuario = "ayudame con informacion"
    respuesta = mensaje_Llega(mensaje_usuario)
    print(respuesta)'''
    app.run(debug=True)
    #devolver_audio("texto")