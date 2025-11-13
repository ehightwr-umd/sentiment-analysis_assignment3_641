#!/usr/bin/env python3

# Import Libraries:
import argparse, pandas as pd, numpy as np, re, random, os, time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import pickle
from utils import set_seeds

# Set Randomness, as required:
set_seeds(42)

# Function for Lowercasing & Removing Punctuation/Special Characters:
def clean_text(text):
    """Lowercase and remove punctuation/special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Function to Tokenize Sentences, Keeping Only Top 10,000 Most Frequent Words:
def preprocess_texts(texts, num_words=10000):
    """Tokenize and convert texts to sequences using Keras Tokenizer."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return sequences, tokenizer

# Function for Generating Sequence Lengths: 25, 50, 100:
def pad_sequences_to_lengths(sequences, lengths=[25, 50, 100]):
    """Pad or truncate sequences to multiple lengths."""
    padded = {}
    for l in lengths:
        padded[l] = pad_sequences(sequences, maxlen=l, padding='post', truncating='post')
    return padded

# Main Function:
def main():
    parser = argparse.ArgumentParser(description="Preprocess IMDB dataset for RNN models")
    parser.add_argument('--input', required=True, help='Path to CSV dataset')
    parser.add_argument('--output_dir', default='data/processed', help='Where to save processed sequences')
    parser.add_argument('--results_file', default='results/preprocessing_times.csv', help='CSV to save timings')
    parser.add_argument('--stats_file', default='results/preprocessing_stats.csv', help='CSV to save vocab/length stats')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    timing = {}
    start_total = time.time()

    # Load the Data:
    start = time.time()
    df = pd.read_csv(args.input)
    end = time.time()
    timing['load_dataset'] = end - start
    print(f"Task: load {len(df)} samples | Time: {timing['load_dataset']:.2f} sec")

    # Clean the Loaded Data:
    start = time.time()
    df['review'] = df['review'].astype(str).apply(clean_text)
    end = time.time()
    timing['clean_text'] = end - start
    print(f"Task: cleaned text | Time: {timing['clean_text']:.2f} sec")

    # Calculate Pre-Processed Review Statistics before Limiting Vocabulary:
    start = time.time()
    tokenized = [r.split() for r in df['review']]
    lengths = [len(tokens) for tokens in tokenized]
    avg_length = np.mean(lengths)
    all_tokens = [token for tokens in tokenized for token in tokens]
    full_vocab = set(all_tokens)
    vocab_size_full = len(full_vocab)

    os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
    pd.DataFrame([{
        "Average_Review_Length": avg_length,
        "Full_Vocabulary_Size": vocab_size_full
    }]).to_csv(args.stats_file, index=False)
    print(f"Pre-processing statistics saved to {args.stats_file}\n")
    end = time.time()
    timing['compute_stats'] = end - start
    print(f"Task: compute pre-processed stats | Time: {timing['compute_stats']:.2f} sec")
    
    # Process Labels (String to Binary):
    start = time.time()
    if df['sentiment'].isin(['positive', 'negative']).all():
        df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    else:
        raise ValueError("Unexpected labels in 'sentiment' column")
    end = time.time()
    timing['process_labels'] = end - start
    print(f"Task: processed labels | Time: {timing['process_labels']:.2f} sec")

    # Generate Training / Testing Datasets:
    start = time.time()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df.iloc[:25000]
    test_df = df.iloc[25000:50000]
    end = time.time()
    timing['split_dataset'] = end - start
    print(f"Task: split dataset | Time: {timing['split_dataset']:.2f} sec")

    # Apply Tokenizer to Train / Test Datasets:
    start = time.time()
    sequences_train, tokenizer = preprocess_texts(train_df['review'].tolist(), num_words=10000)
    sequences_test = tokenizer.texts_to_sequences(test_df['review'].tolist())
    end = time.time()
    timing['tokenize'] = end - start
    print(f"Task: tokenize text | Time: {timing['tokenize']:.2f} sec")

    # Pad Sequence Lengths to [25, 50, 100]:
    start = time.time()
    padded_train = pad_sequences_to_lengths(sequences_train, lengths=[25, 50, 100])
    padded_test = pad_sequences_to_lengths(sequences_test, lengths=[25, 50, 100])
    end = time.time()
    timing['pad_sequences'] = end - start
    print(f"Task: pad sequences | Time: {timing['pad_sequences']:.2f} sec")

    # Save Pre-processed Data
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    # Save: Training Sequences
    for l, seqs in padded_train.items():
        torch.save(torch.tensor(seqs), os.path.join(args.output_dir, f'train_seq{l}.pt'))

    # Save: Testing Sequences
    for l, seqs in padded_test.items():
        torch.save(torch.tensor(seqs), os.path.join(args.output_dir, f'test_seq{l}.pt'))

    # Save: Labels
    torch.save(torch.tensor(train_df['label'].tolist()), os.path.join(args.output_dir, 'train_labels.pt'))
    torch.save(torch.tensor(test_df['label'].tolist()), os.path.join(args.output_dir, 'test_labels.pt'))

    # Save: Tokenizer
    with open(os.path.join(args.output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    end = time.time()
    timing['save_data'] = end - start
    print(f"Task: save processed data | Time: {timing['save_data']:.2f} sec")

    # Save: Time Results
    end_total = time.time()
    timing['total_runtime'] = end_total - start_total
    print(f"Total preprocessing runtime: {timing['total_runtime']:.2f} sec")

    pd.DataFrame([timing]).to_csv(args.results_file, index=False)
    print(f"Timing results saved to {args.results_file}")

    # Final Confirmation of Pre-Processing:
    print(f"Preprocessing complete; Data saved to {args.output_dir}")

if __name__ == "__main__":
    main()
