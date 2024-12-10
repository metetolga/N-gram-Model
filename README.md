
# N-Gram Language Model and Word Suggestion System

This project implements an N-gram language model capable of tokenizing text, handling out-of-vocabulary (OOV) words, calculating perplexity, and suggesting the next word(s) based on input sequences.

## Features

- **Sentence Splitting**: Splits a text dataset into individual sentences.
- **Tokenization**: Converts sentences into lowercased word tokens using the Natural Language Toolkit (NLTK).
- **Vocabulary Handling**:
  - Filters words by minimum frequency to build a closed vocabulary.
  - Replaces out-of-vocabulary words with a placeholder (`<unk>`).
- **N-gram Counting**: Counts unigrams, bigrams, trigrams, or higher-order n-grams in tokenized sentences.
- **Probability Estimation**: Estimates the probability of a word given its preceding n-gram using smoothing.
- **Perplexity Calculation**: Measures the quality of the language model by computing the perplexity of a given sentence.
- **Word Suggestion**: Provides word suggestions based on input tokens and n-gram models, optionally filtered by a starting prefix.

## Prerequisites

- Python 3.x
- Required Libraries:
  - `math`
  - `random`
  - `numpy`
  - `pandas`
  - `collections`
  - `nltk`

## Setup

1. Clone the repository or download the source code.
2. Install the required libraries using `pip`:

   ```bash
   pip install numpy pandas nltk
   ```

3. Download the NLTK tokenizer resource:

   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

4. Place your dataset file (`en_US.twitter.txt`) in the same directory as the script.

## Usage

### Running the Script

1. Load the dataset file `en_US.twitter.txt` by placing it in the working directory.
2. Run the script:

   ```bash
   python script.py
   ```

3. The script will:
   - Split and tokenize the dataset.
   - Preprocess the data to handle OOV words.
   - Train unigram, bigram, and trigram models.
   - Calculate the perplexity of a test sentence.
   - Provide word suggestions based on a sequence of input tokens.

### Example Output

- **Perplexity for a Test Sentence**:
  ```
  Perplexity for test sentence: 123.4567
  ```

- **Word Suggestions**:
  ```
  Suggested words: [('dog', 0.25), ('day', 0.15)]
  ```

## Code Structure

- **Text Preprocessing**:
  - `split_sentences(text_data)`
  - `tokenize_sentences(sentences)`
  - `preprocess_data(train_data, test_data, count_threshold)`

- **Vocabulary Management**:
  - `count_words(tokenized_sentences)`
  - `get_words_min_freq(tokenized_sentences, count_threshold)`
  - `handle_oov_words(tokenized_sentences, vocabulary, unknown_token="<unk>")`

- **N-gram Operations**:
  - `count_n_grams(data, n, start_token='<s>', end_token='</e>')`
  - `estimate_probability(word, previous_n_gram, ...)`
  - `estimate_probabilities(previous_n_gram, ...)`

- **Utilities**:
  - `calculate_perplexity(sentence, ...)`
  - `suggest_a_word(previous_tokens, ...)`
  - `get_suggestions(previous_tokens, ...)`
