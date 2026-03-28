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

nltk.download('punkt', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg





raw_text = gutenberg.raw('melville-moby_dick.txt')







def preprocess(text):

    """Returns lowercase word tokens with no punctuation (using NLTK)."""
    text = text.lower()
    tokens = word_tokenize(text)                    
    tokens = [word for word in tokens if word.isalpha()]   
    return tokens

tokens = preprocess(raw_text)
total_tokens = len(tokens)

vocab = list(set(tokens))
vocab_size = len(vocab)

print("=== Preprocessing ===")
print(f"Vocabulary size: {vocab_size}")
print(f"Total tokens: {total_tokens}")









def build_trigram_model(tokens):

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

    smoothed_probs = {}
    for context, counters in trigram_counts.items():
        total_context = sum(counters.values())
        denominator = total_context + vocab_size
        smoothed_probs[context] = {}
        for word, count in counters.items():
            smoothed_probs[context][word] = (count + 1) / denominator
    return smoothed_probs  


smoothed_probs= laplace_smoothing(trigram_counts, vocab_size)








def generate_text(seed, smoothed_probs, vocab, num_words=30):

    random.seed(42)
    generated = seed[:]
    while len(generated) < num_words:
        context = (generated[-2], generated[-1])
        if context in smoothed_probs:
            next_words = smoothed_probs[context]
            next_word = max(next_words, key=next_words.get)
        else:
            next_word = random.choice(vocab)
        generated.append(next_word)

    return " ".join(generated)


seed = ['sea', 'captain']


generated_text = generate_text(seed, smoothed_probs, vocab, num_words=30)

print("=== Text Generation (Laplace Smoothing) ===")
print(f"Seed: {' '.join(seed)}")
print(f"Generated: {generated_text}")









def compute_perplexity(test_tokens, smoothed_probs, vocab_size):

    if len(test_tokens) < 3:
        return print(f"infinity")
    
    log_sum = 0.0
    N = len(test_tokens) - 2                    
    for i in range(2, len(test_tokens)):
        context = (test_tokens[i-2], test_tokens[i-1])
        w = test_tokens[i]
    
        if context in smoothed_probs and w in smoothed_probs[context]:
            p = smoothed_probs[context][w]
        else:
            
            if context in trigram_counts:
                total_context = sum(trigram_counts[context].values())
            else:
                total_context = 0
            denom = total_context + vocab_size
            p = 1.0 / denom
        log_sum += math.log(p + 1e-10)          

    perplexity = math.exp(-log_sum / N)
    return perplexity







test_sentence = "the king is dead"
test_tokens = preprocess(test_sentence)
perplexity = compute_perplexity(test_tokens, smoothed_probs, vocab_size)

print("=== Perplexity ===")
print("Test sentence: 'the king is dead'")
print(f"Perplexity: {perplexity}")