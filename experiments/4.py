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