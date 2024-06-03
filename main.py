from flask import Flask, request, Response ,jsonify, send_file, send_from_directory
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
lemmatizer = WordNetLemmatizer()
confirming = False
user_name = ''
user_city = ''
pending_payment = False
     
# Cargar datos de intenciones desde un archivo JSON y modelos
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('IntellichatModel.h5')
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

#Limpieza de oraciones
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
    # Asegúrese de que la bolsa de palabras tenga la longitud correcta (142))
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

def generar_link_pago(ciudad, zona=None):
    enlaces_pago = {
        'cartagena': 'https://payp.page.link/D1cbL',
        'panama': 'https://payp.page.link/kkqBA',
        'galapagos_santaCruz': 'https://payp.page.link/Y5boM',
        'galapagos_Bahia_santaCruz': 'https://payp.page.link/hC7sj',
        'default':'Claro!'
    }
    if ciudad.lower() == 'galapagos':
        if zona:
            if zona.lower() == 'santa cruz':
                return enlaces_pago['galapagos_santaCruz']
            elif zona.lower() == 'bahia':
                return enlaces_pago['galapagos_Bahia_santaCruz']
            else:
                return 'Lo siento, no tengo un enlace de pago para esa zona específica de Galápagos.'
        else:
            return '¿Podrías especificar si es para Santa Cruz o Bahía en Galápagos?'
    return enlaces_pago.get(ciudad.lower(), enlaces_pago['default'])

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

def chatbot_response(message, user_id):
    global user_name, user_city, confirmation_info

    # Personalizar el mensaje si el nombre de usuario ya está establecido
    if user_name:
        message = f"{user_name}, {message}"

    # Verificar si el mensaje es suficientemente largo y alfanumérico
    if len(message) < 3 or not any(char.isalnum() for char in message):
        return "Lo siento, no entendí tu pregunta."

    # Clasificar la intención del mensaje
    ints = predict_class(message)
    tag = ints[0]['intent']

    mentioned_countries = ["Panama", "Galapagos", "Cartagena"]
    countries_mentioned = [country for country in mentioned_countries if country.lower() in message.lower()]

    if tag == "saludo":
        return {"mensaje": "¡Hola! Bienvenido a Trivai. ¿Puedes ayudarme con tu nombre?"}

    
    if tag == "forma_pago" or tag == "credito" or tag == "debito":
        if pending_payment and user_city.lower() == 'galapagos':
            enlace_pago = generar_link_pago('galapagos', message)
            pending_payment = False
            return {
                "mensaje": f"Utiliza el siguiente enlace para realizar el pago: {enlace_pago}"
            }
        elif user_city.lower() == 'galapagos':
            pending_payment = True
            return {
                "mensaje": "¿Podrías especificar si es para Santa Cruz o Bahía en Galápagos?"
            }
        else:
            enlace_pago = generar_link_pago(countries_mentioned[0] if countries_mentioned else 'default')
            respuesta = get_response(ints, intents)
            return {
                "mensaje": f"{respuesta} \nEnlace de pago: {enlace_pago}"
            }

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
                
        respuesta = ' '.join(respuestas_pais) if respuestas_pais else "Lo siento, no tengo información sobre esos paquetes."

    if countries_mentioned and (not user_city or countries_mentioned[0].lower() != user_city.lower()):
        user_city = countries_mentioned[0]  # Almacenar la nueva ciudad mencionada
        if user_name:
            return f"¡Hola {user_name}! Vi que estás interesado en {user_city}. ¿Te gustaría más información sobre el paquete?"
        else:
            return f"¡Hola! Vi que estás interesado en {user_city}. ¿Te gustaría más información sobre el paquete?"

    respuesta = get_response(ints, intents)
    if not respuesta:
        respuesta = "Lo siento, no puedo responder esa pregunta en este momento."

    return respuesta

# Routes
@app.route('/chat', methods=['POST'])
def chat_route():
    global user_name, user_city

    data = request.json
    user_id = data.get('user_id', None)  # Extraer user_id si está disponible, de lo contrario, None

    if 'mensaje' not in data:
        return jsonify({"respuesta": "No se proporcionó un mensaje válido."})

    mensaje_recibido = data['mensaje']
    mentioned_countries = ["Panama", "Galapagos", "Cartagena"]
    countries_mentioned = [country for country in mentioned_countries if country.lower() in mensaje_recibido.lower()]

    if not user_name:
        extracted_name = extract_name(mensaje_recibido)
        if extracted_name:
            user_name = extracted_name
            if countries_mentioned and (not user_city or countries_mentioned[0].lower() != user_city.lower()):
                user_city = countries_mentioned[0]
                return jsonify({"respuesta": f"¡Hola {user_name}! Vi que estás interesado en {user_city}. ¿Te gustaría más información sobre el paquete?"})
            return jsonify({"respuesta": f"¡Hola {user_name}! ¿En qué puedo ayudarte hoy?"})
        else:
            if countries_mentioned:
                return jsonify({"respuesta": f"¡Hola! Vi que estás interesado en {countries_mentioned}. ¿Puedes decirme tu nombre para continuar?"})
            return jsonify({"respuesta": "¡Hola! Bienvenido a Trivai. ¿Puedes ayudarme con tu nombre?"})

    respuesta = chatbot_response(mensaje_recibido, user_id)
    return jsonify({"respuesta": respuesta})


    
@app.route('/img/<path:filename>')
def static_files(filename):
    # Asegúrate de que el directorio de imágenes sea correcto
    image_directory = "img"

    try:
        # Intenta abrir la imagen
        return send_file(f"{image_directory}/{filename}", mimetype='image/png')
    except Exception as e:
        # Si hay algún error, devuelve un mensaje de error
        return str(e)

def add_hotel_image(response):
    # Definir una lista de imágenes de hoteles
    hotel_images = [
        "img1.png",
        "img2.png",
    ]
    
    # Si el bot responde con la etiqueta "Brasil", elige una imagen de hotel al azar y devuelve la respuesta con la imagen
    if 'Brasil' in response:
        hotel_image = random.choice(hotel_images)
        image_url = f"/img/{hotel_image}"
        response_with_image = f"{response} <img src='{image_url}' alt='Hotel Image'>"
        return response_with_image
    return response
    
    # Si no se encuentra la etiqueta "Brasil", devolver la respuesta original sin imagen



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