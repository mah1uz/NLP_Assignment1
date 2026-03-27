# import sys
# print(sys.executable)
# activate conda nlp_assignment
#  python ngram_model.py
import nltk
import re
import math
import collections
from collections import defaultdict, Counter
import random


# nltk.download('gutenberg')
from nltk.corpus import gutenberg

# THE CORPUS
raw_text = gutenberg.raw('melville-moby_dick.txt')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) 
    tokens = text.split()
    return tokens

tokens = preprocess(raw_text)
total_tokens = len(tokens)

vocab = list(set(tokens))
vocab_size = len(vocab)



print("=== Preprocessing ===")
print(f"Vocabulary size: {vocab_size}")
print(f"Total tokens: {total_tokens}")


def build_trigram_model(tokens: list[str]) -> dict:
    """Returns trigram_counts[(w1,w2)][next_word] = count."""

    trigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 2):
     
        w1 = tokens[i]        
        w2 = tokens[i + 1]  
        w3 = tokens[i + 2]  
        trigram_counts[(w1, w2)][w3] += 1

    return  dict(trigram_counts)  # return type is dict

trigram_counts = build_trigram_model(tokens)

print(f"Trigram counts: {len(trigram_counts)}")
# print(f"Trigram counts: {trigram_counts}")