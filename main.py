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



app = Flask(__name__)
nlp = spacy.load('es_core_news_sm')
# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
confirming = False
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
# Definir una función para manejar la confirmación de viaje
def handle_confirmation():
    # Inicializar el mensaje de confirmación
    confirmation_message = ""

    # Mostrar el mensaje de confirmación
    for intent in intents['intents']:
        if intent['tag'] == 'confirmacion':
            confirmation_message = random.choice(intent['responses'])
            break

    # Inicializar una lista para almacenar las respuestas del usuario
    user_responses = []

    # Iterar sobre las preguntas y solicitar al usuario que proporcione la información
    for question in confirmation_questions:
        user_response = input(question + " ")
        user_responses.append(user_response)

    # Devolver el mensaje de confirmación y las respuestas del usuario
    return confirmation_message, user_responses

# Obtener la respuesta del usuario al patrón de confirmación de viaje
def get_confirmation_response():
    mensaje_recibido = data['mensaje']
    data = request.json
    # Solicitar y guardar la información del usuario
    global confirmation_response
    confirmation_message, user_responses = handle_confirmation()
    # Formatear las respuestas del usuario para enviarlas como mensaje
    user_responses_message = "\n".join([f"- {question}: {response}" for question, response in zip(confirmation_questions, user_responses)])
    
    # Imprimir la respuesta del usuario por consola
    print("Respuesta del usuario:")
    print(mensaje_recibido)

    # Almacenar las respuestas del usuario en un diccionario
    confirmation_response = {question: response for question, response in zip(confirmation_questions, user_responses)}

    return confirmation_message, user_responses_message

# Definir la lista de preguntas de confirmación
confirmation_questions = [
    "Nombre:",
    "Correo electrónico:",
    "Destino:",
    "Fecha tentativa:",
    "Cuantas personas viajan:",
    "Edades:",
    "Salida de Quito o Guayaquil:"
]

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
    list_of_intents = intents['intents']  # Aquí se cambia de intents_json a intents
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "Lo siento, no puedo responder esa pregunta en este momento."

         
def chatbot_response(mensaje_recibido, user_id):
    mentioned_countries = ["Panamá", "Colombia", "Argentina", "Brasil", "Caribe", "Bahamas", "Cuba", "Peru"]  
    # Obtener los países mencionados
    # Inicializar una lista para almacenar los países mencionados en el mensaje del usuario
    countries_mentioned = []
    for country in mentioned_countries:
        if country.lower() in mensaje_recibido.lower():
            countries_mentioned.append(country)
    # Verificar si se mencionaron países en el mensaje del usuario
    if countries_mentioned:
        # Obtener las intenciones antes de llamar a get_response
        ints = predict_class(mensaje_recibido)
        # Inicializar una lista para almacenar las respuestas de cada país mencionado
        respuestas = []
        for country in countries_mentioned:
            # Llamar a la función get_response con los argumentos apropiados
            respuesta_pais = get_response(ints, intents)
            respuestas.append(respuesta_pais)
        # Combinar las respuestas en un solo mensaje
        respuesta = ' '.join(respuestas)
    else:
        ints = predict_class(mensaje_recibido)
        respuesta = get_response(ints, intents)  # Pasar una lista vacía si no se mencionan países
    return respuesta

    
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
    global user_name
    global confirming
    global confirmation_questions

    data = request.json
    user_id = None

    if 'mensaje' in data:
        mensaje_recibido = data['mensaje']
        nombre = extract_name(mensaje_recibido)

        if 'user_id' in data:
            user_id = data['user_id'] 
        else:
            user_id = None  

        if nombre:
            user_id = nombre
            respuesta = f"¡Bienvenido, {nombre}! ¿En qué puedo ayudarte hoy?"
        else:
            if confirming:
                # Si estamos en el proceso de confirmación, esperamos respuestas a las preguntas de confirmación
                # y luego enviamos un mensaje de confirmación con los datos proporcionados por el usuario
                confirmation_response = {}
                for question in confirmation_questions:
                    respuesta_usuario = data.get(question.lower(), "")
                    confirmation_response[question] = respuesta_usuario

                # Formatear las respuestas del usuario como una lista o con saltos de línea
                user_responses_message = "\n".join([f"- {question}: {response}" for question, response in confirmation_response.items()])

                # Imprimir las respuestas del usuario por consola
                print("Respuestas del usuario:")
                print(json.dumps(confirmation_response, indent=4))

                respuesta = f"Gracias por proporcionar la información:\n{user_responses_message}\n¿Hay algo más en lo que pueda ayudarte?"
                confirming = False  # Salir del modo de confirmación
            else:
                # Si no estamos en el proceso de confirmación, procesamos el mensaje como de costumbre
                ints = predict_class(mensaje_recibido)
                respuesta = get_response(ints, intents)

                if respuesta.startswith("Para poder ayudarte necesitamos saber la siguiente información"):
                    # Si la respuesta del chatbot indica que se necesita más información,
                    # activamos el modo de confirmación y solicitamos las respuestas del usuario
                    confirming = True
                    # Formatear las preguntas de confirmación como una lista o con saltos de línea
                    confirmation_questions_message = "\n".join(confirmation_questions)
                    respuesta = f"{respuesta}\n{confirmation_questions_message}"
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