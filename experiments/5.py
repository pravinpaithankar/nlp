import spacy
from textblob import TextBlob

# Load the spacy model for English
sp = spacy.load("en_core_web_sm")

# Creating a list of positive and negative sentences.
mixed_sen = [
    'This chocolate truffle cake is really tasty',
    'This party is amazing!',
    'My mom is the best!',
    'App response is very slow!',
    'The trip to India was very enjoyable'
]

# An empty list for obtaining the extracted aspects from sentences.
ext_aspects = []

# Performing Aspect Extraction
for sen in mixed_sen:
    important = sp(sen)
    descriptive_item = ''
    target = ''

    for token in important:
        if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
            target = token.text
        if token.pos_ == 'ADJ':
            added_terms = ''
            for mini_token in token.children:
                if mini_token.pos_ != 'ADV':
                    continue
                added_terms += mini_token.text + ' '
            descriptive_item = added_terms + token.text

    ext_aspects.append({'aspect': target, 'description': descriptive_item})

print("ASPECT EXTRACTION\n")
print(ext_aspects)

# Associating Sentiment
for aspect in ext_aspects:
    aspect['sentiment'] = TextBlob(aspect['description']).sentiment

print("\nSENTIMENT ASSOCIATION\n")
print(ext_aspects)

print("")
print("")

import spacy
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from pprint import pprint
import nltk

# Download stopwords
nltk.download('stopwords')

# Load spacy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Sample data for topic modeling
documents = [
    'This chocolate truffle cake is really tasty',
    'The party was amazing and everyone enjoyed it!',
    'My mom is the best and she loves me so much',
    'The app response is very slow, and it frustrates me',
    'The trip to India was very enjoyable and the experience was unforgettable',
]

# 1. Preprocessing (tokenization, stopwords removal, lemmatization)
stop_words = set(stopwords.words('english'))

def preprocess(doc):
    # Tokenize and lemmatize
    doc = nlp(doc)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# 2. Create Dictionary and Corpus
# Create a dictionary representation of the documents
id2word = corpora.Dictionary(processed_docs)

# Create the Bag of Words corpus
corpus = [id2word.doc2bow(text) for text in processed_docs]

# 3. Applying LDA Model (Topic Modeling)
lda_model = gensim.models.LdaMulticore(corpus, id2word=id2word, num_topics=3, passes=10, workers=2, random_state=42)

# 4. Output the topics
pprint(lda_model.print_topics())

# Show the topic distribution for each document
for i, topic_distribution in enumerate(lda_model[corpus]):
    print(f"\nDocument {i + 1} Topic Distribution:")
    print(topic_distribution)



import spacy
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from pprint import pprint
import nltk

# Download stopwords
nltk.download('stopwords')

# Load spacy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Sample data for topic modeling
documents = [
    'This chocolate truffle cake is really tasty',
    'The party was amazing and everyone enjoyed it!',
    'My mom is the best and she loves me so much',
    'The app response is very slow, and it frustrates me',
    'The trip to India was very enjoyable and the experience was unforgettable',
]

# 1. Preprocessing (tokenization, stopwords removal, lemmatization)
stop_words = set(stopwords.words('english'))

def preprocess(doc):
    # Tokenize and lemmatize
    doc = nlp(doc)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

# 2. Create Dictionary and Corpus
# Create a dictionary representation of the documents
id2word = corpora.Dictionary(processed_docs)

# Create the Bag of Words corpus
corpus = [id2word.doc2bow(text) for text in processed_docs]

# 3. Applying LDA Model (Topic Modeling)
lda_model = gensim.models.LdaMulticore(corpus, id2word=id2word, num_topics=3, passes=10, workers=2, random_state=42)

# 4. Output the topics
pprint(lda_model.print_topics())

# Show the topic distribution for each document
for i, topic_distribution in enumerate(lda_model[corpus]):
    print(f"\nDocument {i + 1} Topic Distribution:")
    print(topic_distribution)


# PRACTICAL 5 : ASPECT MINING AND TOPIC MODELING

# Aim

# The aim of this project is to implement aspect mining and topic
# modeling techniques for extracting meaningful insights from textual
# data. By identifying key aspects and underlying topics, we can better
# understand the sentiments and themes present in the text.



# Objectives

# 1. To utilize Natural Language Processing (NLP) techniques for
# extracting aspects from text data.

# 2. To implement topic modeling using algorithms like LDA to discover
# latent topics in the dataset.

# 3. To evaluate the effectiveness of the aspect mining and topic
# modeling techniques through qualitative analysis of results.



# Theory

# Aspect mining focuses on identifying specific features or components
# within a given text that are of interest, such as product attributes or
# sentiment-related topics. Topic modeling is a statistical method that
# analyzes text to uncover hidden thematic structures, helping to group
# similar content and facilitate better information retrieval and
# understanding.


