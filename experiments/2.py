pos_words = []
neg_words = []

with open('sentimentanalysis.txt') as file:
    for line in file:
        line_attrib = line.strip().split()  # Strip whitespace and split
       
        # Check if line has enough parts
        if len(line_attrib) < 2:
            continue
        
        # Safely extract word and polarity
        try:
            word = line_attrib[0].split('=')[1]
            polarity = line_attrib[1].split('=')[1]
        except IndexError:
            continue

        # Print extracted values for verification

        if polarity == 'positive':
            pos_words.append(word)
        elif polarity == 'negative':
            neg_words.append(word)

# Print only the total counts of positive and negative words
print('Total positive words:', len(pos_words))
print('Total negative words:', len(neg_words))

with open('pos_words.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(pos_words))

with open('neg_words.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(neg_words))



# PRACTICAL 2 : SENTIMENT ANALYSIS

# Aim:

# The aim of this assignment is to implement a sentiment analysis
# technique for classifying textual data into positive, negative, or
# neutral sentiment categories. This classification will help in
# understanding public opinion and emotional tone in various textual
# datasets.



# Objectives:

# 1. To define sentiment analysis and its relevance in understanding
# textual emotions and opinions.

# 2. To explore various sentiment analysis algorithms and libraries used
# for text classification.

# 3. To develop a sentiment analysis model and evaluate its
# performance on a sample dataset.



# Theory:

# Sentiment analysis is a computational technique used to determine
# the emotional tone behind a body of text. By classifying text into
# positive, negative, or neutral sentiments, it provides valuable insights
# into opinions, attitudes, and emotions expressed in social media,
# reviews, and other textual data sources. Various methods, including
# machine learning and deep learning techniques, are employed to
# build effective sentiment analysis models.
