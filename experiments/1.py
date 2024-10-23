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