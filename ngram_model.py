# import sys
# print(sys.executable)
# activate conda nlp_assignment
#  python ngram_model.py
import nltk
import math
import collections
from collections import defaultdict, Counter
import random

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('gutenberg', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg


#for better visibility there are 5-7 lines of gaps are given


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

    trigram_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - 2):
        w1,w2,w3 = tokens[i],tokens[i + 1],tokens[i + 2]
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
        denom = total_context + vocab_size
        smoothed_probs[context] = {}
        for word, count in counters.items():
            smoothed_probs[context][word] = (count + 1) / denom
    return smoothed_probs  


smoothed_probs= laplace_smoothing(trigram_counts, vocab_size)
#print(f"Smoothed probabilities: {smoothed_probs}")








def generate_text(seed, smoothed_probs, vocab, num_words=30):

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


seed = ['white', 'whale']


generated_text = generate_text(seed, smoothed_probs, vocab, 30)

print("=== Text Generation (Laplace Smoothing) ===")
print(f"Seed: {' '.join(seed)}")
print(f"Generated: {generated_text}")









def compute_perplexity(test_tokens, smoothed_probs, vocab_size):

    if len(test_tokens) < 3:
        return float('inf')
    
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







test_sentence = "the captain chased the white whale"
test_tokens = preprocess(test_sentence)
perplexity = compute_perplexity(test_tokens, smoothed_probs, vocab_size)

print("=== Perplexity ===")
print("Test sentence: 'the captain chased the white whale'")
print(f"Perplexity: {perplexity:.4f}")


# bonus part










def build_unigram_model(tokens):

    return dict(Counter(tokens))

def build_bigram_model(tokens):

    bigram_counts = defaultdict(Counter)
    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        bigram_counts[w1][w2] += 1
    return {context: dict(counter) for context, counter in bigram_counts.items()}










def interpolated_probability(context, word, trigram_probs,
                             bigram_probs, unigram_probs,
                             vocab_size, lambdas=(0.6, 0.3, 0.1)):

    λ3, λ2, λ1 = lambdas

    p3 = 1.0 / vocab_size
    if context in trigram_probs:
        tri_total = sum(trigram_probs[context].values())
        tri_cnt = trigram_probs[context].get(word, 0)
        p3 = (tri_cnt + 1) / (tri_total + vocab_size)

    p2 = 1.0 / vocab_size
    if len(context) == 2:
        w2 = context[1]
        if w2 in bigram_probs:
            bi_total = sum(bigram_probs[w2].values())
            bi_cnt = bigram_probs[w2].get(word, 0)
            p2 = (bi_cnt + 1) / (bi_total + vocab_size)

    uni_total = sum(unigram_probs.values())
    p1 = (unigram_probs.get(word, 0) + 1) / (uni_total + vocab_size)

    return ((λ3 * p3) + (λ2 * p2) + (λ1 * p1))









def generate_text_interpolated(seed, trigram_probs, bigram_probs, 
                               unigram_probs, vocab, vocab_size, num_words=30):
  
    generated = seed[:]

    while len(generated) < num_words:
        if len(generated) >= 2:
            context = (generated[-2], generated[-1])
        elif len(generated) == 1:
            context = (generated[-1],)   # for bigram fallback
        else:
            context = tuple()

        # Greedy: find word with highest interpolated probability
        best_word = None
        best_prob = -1.0
        for w in vocab:
            prob = interpolated_probability(context, w, trigram_probs, 
                                            bigram_probs, unigram_probs, vocab_size)
                                            
            if prob > best_prob:
                best_prob = prob
                best_word = w

        generated.append(best_word)

    return " ".join(generated)








unigram_counts = build_unigram_model(tokens)
bigram_counts  = build_bigram_model(tokens)

# === Generate story with interpolation (same seed as Laplace) ===
seed = ['white', 'whale']                     # you can change to ['the', 'king']
interpolated_text = generate_text_interpolated(seed, trigram_counts, bigram_counts, unigram_counts,
                                               vocab, vocab_size, num_words=30) 
    
    





test_sentence = "the captain chased the white whale"
test_tokens = preprocess(test_sentence)




def compute_perplexity_interpolated(test_tokens, trigram_probs, bigram_probs, unigram_probs, vocab_size):

    if len(test_tokens) < 3:
        return float('inf')

    lg = 0.0
    n = len(test_tokens) - 2

    for i in range(2, len(test_tokens)):
        ctx = (test_tokens[i-2], test_tokens[i-1])
        w = test_tokens[i]
        prob = interpolated_probability(ctx, w, trigram_probs, bigram_probs, unigram_probs, vocab_size)
        lg += math.log(prob + 1e-10)

    return math.exp(-lg / n)






interpolated_perplexity = compute_perplexity_interpolated(
    test_tokens, trigram_counts, bigram_counts, unigram_counts, vocab_size
)



#printing all
print("\n=== Bonus (if attempted) ===")
print(f"Generated (Interpolation): {interpolated_text}")
print(f"Perplexity (Interpolation): {interpolated_perplexity:.4f}")
print("Comparison: The interpolated model generates a clean text than pure Laplace smoothing.")