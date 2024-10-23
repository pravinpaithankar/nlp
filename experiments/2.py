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
