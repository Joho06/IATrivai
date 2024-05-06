from googletrans import Translator

def translate_file(input_file, output_file, source_lang='en', target_lang='es'):
    translator = Translator()

    # Leer el contenido del archivo de entrada
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Traducir y escribir el contenido traducido en el archivo de salida
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in lines:
            translation = translator.translate(line, src=source_lang, dest=target_lang)
            if translation is not None:  # Verificar si la traducción no es None
                file.write(translation.text + '\n')
            else:
                file.write('Error de traducción\n')

# Traducir el archivo test.txt al español
translate_file('test.txt', 'test_es.txt', source_lang='en', target_lang='es')
