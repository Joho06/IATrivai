# Configuración del Entorno y Ejecución

## Creación del Entorno Virtual
```
py -m venv entornoMV
cd entornoMV
cd Scripts 
activate 
```
## Actualización de Python y PIP 
```
sudo apt install python3-pip
```
## Instalación de las dependencias 
```
pip install --upgrade pip
pip install google-auth
pip install google-cloud-speech
pip install google-cloud-texttospeech
pip install Flask
pip install pydub
```
## Inicialización del servidor
```
py main.py
```
