# Importing necessary libraries
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
import sentencepiece as spm
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Sample text
text = "Natural language processing (NLP) is a crucial technology for modern applications like chatbots, translation, and AI."

# 1. Basic Word Tokenization (NLTK)
print("Basic Word Tokenization (NLTK):")
word_tokens = word_tokenize(text)
print(word_tokens)

# 2. Subword Tokenization (BERT's WordPiece Tokenizer)
print("\nSubword Tokenization (BERT - WordPiece):")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tokenizer.tokenize(text)
print(bert_tokens)

# 3. Byte Pair Encoding (BPE) with GPT-2 Tokenizer
print("\nByte Pair Encoding (GPT-2):")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokens = gpt2_tokenizer.tokenize(text)
print(gpt2_tokens)

# 4. SentencePiece Tokenization (Pretrained on RoBERTa)
print("\nSentencePiece Tokenization (RoBERTa):")
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_tokens = roberta_tokenizer.tokenize(text)
print(roberta_tokens)

# 5. Train your own SentencePiece tokenizer (for custom data)
print("\nCustom SentencePiece Tokenizer (Training):")

# You would typically train on large text data, but here we simulate with small data
sample_data = "Natural language processing is essential for modern AI applications."
with open("sample_text.txt", "w") as f:
    f.write(sample_data)

# Train SentencePiece model with smaller vocabulary size
spm.SentencePieceTrainer.Train('--input=sample_text.txt --model_prefix=m --vocab_size=28')
sp = spm.SentencePieceProcessor(model_file='m.model')

# Tokenize using custom SentencePiece model
sentencepiece_tokens = sp.encode_as_pieces(text)
print(sentencepiece_tokens)
