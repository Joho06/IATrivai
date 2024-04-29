from google.cloud import texttospeech

class TextToSpeech:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    def synthesize_speech(self, text, output_file_path="audiosMp3/output.mp3"):
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-US",
            name="es-US-Neural2-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_file_path, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{output_file_path}"')
