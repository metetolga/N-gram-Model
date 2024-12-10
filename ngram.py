import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
nltk.data.path.append('.')
nltk.download('punkt_tab')

# Load the dataset
with open('en_US.twitter.txt') as f:
    data = f.read()

# Display data type, length, and a preview
print("Data type:", type(data))
print("Number of characters:", len(data))
print("First 300 characters of the data:")
print(data[:300])
print("-------")
print("Last 300 characters of the data:")
print(data[-300:])
print("-------")

# Function to split data into sentences
def split_sentences(text_data):
    sentences = text_data.split('\n')
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]
    return sentences

# Tokenize sentences into words
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences

# Preprocessing function for tokenized data
def preprocess_data(train_data, test_data, count_threshold):
    vocabulary = get_words_min_freq(train_data, count_threshold)
    train_data_replaced = handle_oov_words(train_data, vocabulary)
    test_data_replaced = handle_oov_words(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary

# Tokenize and preprocess the data
def get_tokenized_data(text_data):
    sentences = split_sentences(text_data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

# Function to count word frequencies in tokenized sentences
def count_words(tokenized_sentences):
    word_counter = defaultdict(int)
    for sentence in tokenized_sentences:
        for word in sentence:
            word_counter[word] += 1
    return word_counter

# Filter words by frequency
def get_words_min_freq(tokenized_sentences, count_threshold):
    closed_vocab = []
    word_counter = count_words(tokenized_sentences)
    for word, count in word_counter.items():
        if count >= count_threshold:
            closed_vocab.append(word)
    return closed_vocab

# Handle Out Of Vocabulary (OOV) words by replacing them with <unk>
def handle_oov_words(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocab = set(vocabulary)
    new_sentences = []
    for sentence in tokenized_sentences:
        new_sentence = [word if word in vocab else unknown_token for word in sentence]
        new_sentences.append(new_sentence)
    return new_sentences

# Count n-grams in a dataset
def count_n_grams(data, n, start_token='<s>', end_token='</e>'):
    n_grams = defaultdict(int)
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i + n]
            n_grams[n_gram] += 1
    return n_grams

# Estimate the probability of a word given its previous n-gram
def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocab_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    prev_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denom = prev_n_gram_count + k * vocab_size

    nplus1_gram = previous_n_gram + (word,)
    nplus1_gram_count = n_plus1_gram_counts.get(nplus1_gram, 0)

    nom = nplus1_gram_count + k
    return nom / denom

# Estimate the probabilities for all possible next words
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    vocab_size = len(vocabulary) + 2  # including <e> and <unk>
    probabilities = {}
    
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocab_size, k=k)
        probabilities[word] = probability

    return probabilities

# Create a count matrix for n-grams
def get_count_matrix(n_plus1_gram_counts, vocabulary):
    vocabulary = vocabulary + ["<e>", "<unk>"]
    n_grams = list(set(n_gram[:-1] for n_gram in n_plus1_gram_counts.keys()))
    
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    col_index = {word: j for j, word in enumerate(vocabulary)}
    
    count_matrix = np.zeros((len(n_grams), len(vocabulary)))
    
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

# Create a probability matrix from n-gram counts
def make_probability_matrix(n_plus1_gram_counts, vocabulary, k=1.0):
    count_matrix = get_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

# Calculate the perplexity of a sentence based on n-gram probabilities
def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocab_size, k=1.0):
    n = len(next(iter(n_gram_counts)))  # length of the n-gram
    sentence = ["<s>"] * n + sentence + ["<e>"]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0
    
    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocab_size, k)
        product_pi *= 1 / probability
    
    perplexity = product_pi ** (1 / float(N))
    return perplexity

# Function to suggest the next word given a sequence of previous tokens
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    n = len(next(iter(n_gram_counts)))  # length of the n-gram
    previous_n_gram = previous_tokens[-n:]
    
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)
    
    suggestion = None
    max_prob = 0
    
    for word, prob in probabilities.items():
        if start_with and not word.startswith(start_with):
            continue
        if prob > max_prob:
            suggestion = word
            max_prob = prob

    return suggestion, max_prob

# Get suggestions using different n-gram models (unigram, bigram, trigram, etc.)
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    suggestions = []
    for i in range(len(n_gram_counts_list) - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    tokenized_data = get_tokenized_data(data)
    random.seed(87)
    random.shuffle(tokenized_data)

    # Split into training and testing sets (80% train, 20% test)
    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[:train_size]
    test_data = tokenized_data[train_size:]

    # Preprocess data by setting a minimum frequency threshold (e.g., 2)
    min_freq = 2
    train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, test_data, min_freq)

    # Count n-grams
    unigram_counts = count_n_grams(train_data_processed, 1)
    bigram_counts = count_n_grams(train_data_processed, 2)
    trigram_counts = count_n_grams(train_data_processed, 3)

    # Calculate perplexity for a test sentence
    test_sentence = ['i', 'like', 'a', 'dog']
    perplexity_test = calculate_perplexity(test_sentence, unigram_counts, bigram_counts, len(vocabulary), k=1.0)
    print(f"Perplexity for test sentence: {perplexity_test:.4f}")

    # Get word suggestions based on previous tokens
    previous_tokens = ['i', 'like']
    suggestions = get_suggestions(previous_tokens, [unigram_counts, bigram_counts, trigram_counts], vocabulary, k=1.0, start_with="d")
    print("Suggested words:", suggestions)