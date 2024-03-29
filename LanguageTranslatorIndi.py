# Use a pipeline as a high-level helper
from transformers import pipeline

# model_checkpoint = "ai4bharat/indictrans2-en-indic-1B"
#
# translator = pipeline("translation", model=model_checkpoint, trust_remote_code=True)
#
# translator("How are you?")

sentence = "This repository contains the code and other resources for the paper published at ACL 2023."


model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
translator = pipeline("translation", model=model_checkpoint)

output = translator(sentence)
print(output)
#
#
# # [{'translation_text': 'Comment allez-vous ?'}]
#
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mr")

output = pipe(sentence)
print(output)

from googletrans import Translator

sentence = "This repository contains the code and other resources for the paper published at ACL 2023."

translator = Translator()
translated = translator.translate(sentence, dest='mr')

print(translated.text)

translated = translator.translate(sentence, dest='hi')

print(translated.text)