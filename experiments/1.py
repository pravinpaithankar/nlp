import spacy
from spacy import displacy

# Load the small English NLP model
nlp = spacy.load("en_core_web_sm")

def spacy_ner(text):
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    doc = nlp(text)

    entities = []
    labels = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            entities.append(ent)
            labels.append(ent.label_)

    return entities, labels

def fit_ner(texts):
    """Accepts a list of text strings instead of a DataFrame."""
    print('Fitting Spacy NER model...')

    ner = [spacy_ner(text) for text in texts]
    ner_org = {}
    ner_per = {}
    ner_gpe = {}

    for x in ner:
        for entity, label in zip(x[0], x[1]):
            if label == 'ORG':
                ner_org[entity.text] = ner_org.get(entity.text, 0) + 1
            elif label == 'PERSON':
                ner_per[entity.text] = ner_per.get(entity.text, 0) + 1
            elif label == 'GPE':
                ner_gpe[entity.text] = ner_gpe.get(entity.text, 0) + 1

    return {'ORG': ner_org, 'PER': ner_per, 'GPE': ner_gpe}

# Example static data (list of strings)
texts = [
    "Apple is looking to buy a startup in the United Kingdom.",
    "Elon Musk is the CEO of Tesla.",
    "Google is headquartered in Mountain View, California.",
    "The United States is a large country."
]

# Run the fit_ner function on the static data
named_entities = fit_ner(texts)

# Print the results
print("Organization Named Entities:", named_entities['ORG'])
print("Person Named Entities:", named_entities['PER'])
print("Geopolitical Entity Named Entities:", named_entities['GPE'])



# PRACTICAL 1 : NER

# Aim:

# The aim of this assignment is to explore the use of Named Entity
# Recognition (NER) as an information extraction technique in natural
# language processing. It focuses on identifying and classifying key
# entities in text data to enhance data analysis and understanding.



# Objectives:

# 1. To define Named Entity Recognition and its significance in natural
# language processing.

# 2. To evaluate various NER algorithms and models available for entity
# extraction.

# 3. To implement a practical NER application using a suitable
# programming language and library.



# Theory:

# Named Entity Recognition (NER) is a crucial technique in information
# extraction that identifies and classifies named entities in
# unstructured text into predefined categories, such as persons,
# organizations, locations, dates, and more. By leveraging NER,
# researchers and practitioners can extract meaningful insights from
# large datasets, enabling more efficient data processing and analysis.
