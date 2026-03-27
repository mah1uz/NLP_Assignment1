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


def build_trigram_model(tokens):
    """Returns trigram_counts[(w1,w2)][next_word] = count."""

    trigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 2):
     
        w1 = tokens[i]        
        w2 = tokens[i + 1]  
        w3 = tokens[i + 2]  
        trigram_counts[(w1, w2)][w3] += 1

    return {
        context: dict(counter)
        for context,counter  in trigram_counts.items()
    }

trigram_counts = build_trigram_model(tokens)

# print(f"Trigram counts: {len(trigram_counts)}")
# print(f"Trigram counts: {trigram_counts}")

def laplace_smoothing(trigram_counts, vocab_size):
    smoothed_probs ={}
    context_denominators= {}

    for context, counters in trigram_counts.items():
        total_context = sum(counters.values())
        denominator= total_context + vocab_size
        context_denominators[context] = denominator
        smoothed_probs[context] = {}
        for word, count in counters.items():
            smoothed_probs[context][word] = (count + 1) / denominator
        
    smoothed_probs["context_denominators"] = context_denominators
    return smoothed_probs  
