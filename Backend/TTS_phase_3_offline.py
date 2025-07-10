#python TTS_phase_3_offline.py
#Downloads/testing/bin/speaker_diarisation/Test_convo
# Initialize the TTS engine
from gtts import gTTS

def main_TTS(podcast_text, final_speech_file):
    # language is English (en)
    tts = gTTS(text=podcast_text, lang='en')

    # Save to file (MP3 format)
    tts.save(final_speech_file)

#to check list of models, tts --list_models in cmd with conda environment diarisation activated
