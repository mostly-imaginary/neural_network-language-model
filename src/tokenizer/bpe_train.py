from collections import Counter, defaultdict
from tqdm import tqdm

# Still needs some optimazation but it is useable on the entire wikipedia subset in a reasonable time (a two hours on my machine).

EOW = "\uE000" # End of Word symbol, a special character to denote the end of a word

def word_to_chars(word):
    return list(word) + [EOW]

def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.readlines()

def build_corpus(text):
    corpus = Counter()

    for line in text:
        for word in line.strip().split():
            chars = tuple(word_to_chars(word))
            corpus[chars] += 1

    return corpus

def get_pair_stats(tokens):

    pair_freq = Counter()

    for word in tokens:
        w = word
        for i in range(len(w) - 1):
            pair_freq[(w[i], w[i+1])] += 1

    return pair_freq

def merge_pair(corpus, pair):
    i1, i2 = pair

    if i1 == EOW or i2 == EOW:
        return corpus
    
    new_corpus = Counter()

    for word, freq in corpus.items():

        # skip fast
        if i1 not in word:
            new_corpus[word] += freq
            continue

        new_word = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and word[i] == i1 and word[i+1] == i2:
                new_word.append(i1 + i2)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_corpus[tuple(new_word)] += freq

    return new_corpus

def train_bpe(corpus, vocab_size):

    merges = []

    pair_freq = get_pair_stats(corpus)

    for iteration in tqdm(range(vocab_size), desc="Training BPE"):

        candidates = [p for p in pair_freq if EOW not in p]

        if not candidates:
            break

        best_pair = max(candidates, key=lambda p: pair_freq[p])

        corpus = merge_pair(corpus, best_pair)
        merges.append(best_pair)

        pair_freq = get_pair_stats(corpus)

    return merges

def save_merges(merges, path="NOT_USED_merges.txt"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

def build_vocab(merges, base_chars, special_tokens=None):
    vocab = set(base_chars)

    for a, b in merges:
        if a != EOW and b != EOW:
            vocab.add(a + b)

    if special_tokens:
        vocab.update(special_tokens)

    return sorted(vocab)

def save_vocab(vocab, path="NOT_USED_vocab.txt"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for i, token in enumerate(vocab):
            f.write(f"{token}\t{i}\n")


def train(dataset_path="datasets/wikipedia/v1/wiki_subset.txt", vocab_size=2000, merges_path="NOT_USED_merges.txt", vocab_path="NOT_USED_vocab.txt"):
    print("Loading corpus...")

    text = load_text(dataset_path)
    corpus = build_corpus(text)

    print("Corpus loaded.")

    print("Training BPE...")

    merges = train_bpe(corpus, vocab_size=vocab_size)

    print("BPE training completed.")

    print("Saving merges and vocabulary...")

    save_merges(merges, path=merges_path)
    base_chars = set(ch for word in corpus for ch in word)
    special_tokens = ["<unk>"]
    vocab = build_vocab(merges, base_chars=base_chars, special_tokens=special_tokens)
    save_vocab(vocab, path=vocab_path)

    print("Merges and vocabulary saved.")