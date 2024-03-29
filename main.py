import streamlit as st
from streamlit_mic_recorder import speech_to_text
# from pydub import AudioSegment
# from pydub.playback import play
from transformers import pipeline
from googletrans import Translator
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM


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
    return translatedOutput[0]


def getTextAndConvertToMarathi(input_text="Hello Everyone. We are student of YCCE College."):
    model_checkpoint = "Helsinki-NLP/opus-mt-en-mr"

    translator = pipeline("translation", model=model_checkpoint)
    translation = Translator()
    translatedOutput = translator(input_text)
    translatedMarOutput = translation.translate(input_text, dest='mr')

    print("translatedOutput Marathi - ", translatedMarOutput)
    return translatedOutput[0]['translation_text']
    # return translatedMarOutput.text


def main():
    st.title("Language Translator")
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    if text:
        st.session_state.inputText = text

    query = st.text_area('English Language Input', 'Enter your Text', key="inputText")

    if st.button("Convert"):
        print("type(query) -", type(query))
        pred = getTextAndConvertToHindi(query)
        # pred = getTextAndConvert(query)
        print("Translated hindi output -", pred)

        marPred = getTextAndConvertToMarathi(query)
        print("Translated marathi output -", marPred)

        st.text_area('Converted Language Output - Hindi', pred['translation_text'])
        # st.text_area('Converted Language Output - Marathi ', marPred['translation_text'])
        st.text_area('Converted Language Output - Marathi ', marPred)


if __name__ == '__main__':
    main()

