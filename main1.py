from gtts import gTTS
from playsound import playsound
import streamlit as st
from streamlit_mic_recorder import speech_to_text
# from pydub import AudioSegment
# from pydub.playback import play
from transformers import pipeline
from googletrans import Translator
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from io import BytesIO


def getTextAndConvert(input_text="Hello Everyone. We are student of YCCE College."):
    model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")

    # input_text = "Hello Everyone. We are from YCCE."

    tokenized = tokenizer([input_text], return_tensors='np')
    out = model.generate(**tokenized, max_length=128)
    print(out)

    with tokenizer.as_target_tokenizer():
        translatedOutput = tokenizer.decode(out[0], skip_special_tokens=True)
        print("translatedOutput - ", translatedOutput)
        return translatedOutput


def getTextAndConvertToHindi(input_text="Hello Everyone. We are student of YCCE College."):
    model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"

    translator = pipeline("translation", model=model_checkpoint)

    translatedOutput = translator(input_text)

    print("translatedOutput Hindi - ", translatedOutput)
    return translatedOutput[0]['translation_text']


def getTextAndConvertToMarathi(input_text="Hello Everyone. We are student of YCCE College."):
    model_checkpoint = "Helsinki-NLP/opus-mt-en-mr"

    # translator = pipeline("translation", model=model_checkpoint)
    translation = Translator()
    # translatedOutput = translation(input_text, dest='mr')
    translatedMarOutput = translation.translate(input_text, dest='mr')

    print("translatedOutput Marathi - ", translatedMarOutput)
    # return translatedOutput[0]['translation_text']
    return translatedMarOutput.text


def hinTextToAudio(mytext):
    tts = gTTS(text=mytext, lang='hi')
    print('hindi tts type - ', type(tts))
    tts.save("hinAudio.mp3")
    playsound("hinAudio.mp3")
    return tts


def marTextToAudio(mytext):
    tts = gTTS(text=mytext, lang='mr')
    print('marathi tts type - ', type(tts))
    # tts.save("marAudio.mp3")
    # playsound("marAudio.mp3")
    return tts.text


def hindi_text_to_speech(trans_text, tld="co.in"):
    tts = gTTS(trans_text, lang="hi")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()


def marathi_text_to_speech(trans_text, tld="co.in"):
    tts = gTTS(trans_text, lang="mr")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.read()


def main():
    # hinPred = getTextAndConvertToHindi()
    # hinTextToAudio(hinPred['translation_text'])

    # marPred = getTextAndConvertToMarathi()
    # marTextToAudio(marPred)
    with st.container(border=True):
        st.title("Language Translator")
        text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        if text:
            st.session_state.inputText = text

        query = st.text_area('English Language Input', 'Enter your Text', key="inputText")

        if st.button("Convert"):
            print("type(query) -", type(query))
            hinPred = getTextAndConvertToHindi(query)
            # pred = getTextAndConvert(query)
            print("Translated hindi output -", hinPred)
            with st.container(border=True):
                if hinPred:
                    st.markdown("Converted Language Output - Hindi")
                    st.text_area('Hindi Text', hinPred)
                    st.audio(hindi_text_to_speech(hinPred), format="audio/mp3", start_time=0)

            marPred = getTextAndConvertToMarathi(query)
            print("Translated marathi output -", marPred)
            with st.container(border=True):
                if marPred:
                    st.markdown("Converted Language Output - Marathi")
                    st.text_area('Marathi Text', marPred)
                    st.audio(marathi_text_to_speech(marPred), format="audio/mp3", start_time=0)

            # st.text_area('Converted Language Output - Marathi ', marPred['translation_text'])


if __name__ == '__main__':
    main()
