from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer for English to Marathi translation
model_name = 'Helsinki-NLP/opus-mt-en-mr'  # Model for English to Marathi translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, src_lang='en', tgt_lang='mr'):
    # Prepare the text input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation using the model
    translated_tokens = model.generate(**inputs)
    # Decode the tokens to get the translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Sample text for translation
text_to_translate = "Hello, how are you?"

# Translate the text from English to Marathi
translated_text = translate_text(text_to_translate)

print(f"Original Text: {text_to_translate}")
print(f"Translated Text: {translated_text}")



PRACTICAL 4 : MACHINE TRANSLATION

# Aim:

# The aim of this assignment is to implement a simple machine
# translation system that can convert text from one language to
# another. This will demonstrate the fundamental principles of Natural
# Language Processing (NLP) in translating linguistic content.



# Objectives:

# 1. To study the basic concepts of machine translation, including rule-
# based and statistical methods.

# 2. To explore various NLP libraries and frameworks that facilitate
# machine translation tasks.

# 3. To develop a prototype translation model and evaluate its accuracy
# and efficiency using a sample dataset.



# Theory:

# Machine translation is a subfield of Natural Language Processing
# (NLP) focused on automatically converting text from one language to
# another. Traditional approaches include rule-based methods, which
# rely on linguistic rules, and statistical methods that leverage bilingual
# text corpora for translation. Recent advancements involve neural
# machine translation, utilizing deep learning techniques to improve
# translation quality and fluency. By implementing a simple translation
# model, one can understand the complexities involved in aligning
# languages and the challenges of preserving meaning across linguistic
# boundaries.

