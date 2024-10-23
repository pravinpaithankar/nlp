import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
doc1 = """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes
code readability with the use of significant indentation.
Python is dynamically-typed and garbage-collected. It supports multiple
programming paradigms, including
structured (particularly procedural), object-oriented, and functional
programming. It is often described
as a "batteries included" language due to its comprehensive standard
library.
Guido van Rossum began working on Python in the late 1980s as a
successor to the ABC programming
language and first released it in 1991 as Python 0.9.0. Python 2.0 was
released in 2000 and introduced new
features such as list comprehensions, cycle-detecting garbage collection,
reference counting, and Unicode
support.
Python 3.0, released in 2008, was a major revision that is not completely
backward-compatible with earlier
versions. Python 2 was discontinued with version 2.7.18 in 2020."""

# Process the text with spaCy
docx = nlp(doc1)

# Tokenization and word frequency calculation
stopwords = list(STOP_WORDS)
word_frequencies = {}
for word in docx:
    if word.text.lower() not in stopwords and word.text not in punctuation:
        if word.text.lower() not in word_frequencies:
            word_frequencies[word.text.lower()] = 1
        else:
            word_frequencies[word.text.lower()] += 1

# Normalize word frequencies by dividing by the max frequency
max_freq = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word] / max_freq

# Sentence scoring based on word frequencies
sentence_scores = {}
for sent in docx.sents:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores:
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]

# Selecting the top 30% of sentences based on their scores
from heapq import nlargest
select_length = int(len(sentence_scores) * 0.3)
summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

# Join the selected sentences into a final summary
final_summary = [word.text for word in summary]
summary = ' '.join(final_summary)

print("Original Text Length:", len(doc1.split()))
print("Summary Length:", len(summary.split()))
print("\nSummary:\n", summary)