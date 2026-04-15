def load_vocab(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            token, idx = line.rstrip("\n").split("\t")
            vocab[token] = int(idx)
    return vocab

def load_merges(path):
    merges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            merges.append((a, b))
    return merges

def build_merge_ranks(merges):
    return {pair: i for i, pair in enumerate(merges)}

EOW = "\uE000"

def word_to_chars(word):
    return list(word) + [EOW]

def get_pairs(word):
    return {(word[i], word[i+1]) for i in range(len(word)-1)}

def merge_word(word, merge_ranks):
    word = list(word)

    while True:
        pairs = get_pairs(word)

        # find best merge available
        best_pair = None
        best_rank = float("inf")

        for p in pairs:
            if p in merge_ranks and merge_ranks[p] < best_rank:
                best_pair = p
                best_rank = merge_ranks[p]

        if best_pair is None:
            break

        a, b = best_pair
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word)-1 and word[i] == a and word[i+1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        word = new_word

    return word

def encode(text, vocab, merges):
    merge_ranks = build_merge_ranks(merges)

    tokens = []

    for word in text.strip().split():
        chars = word_to_chars(word)
        pieces = merge_word(chars, merge_ranks)

        print(f"Word: {word} -> Pieces: {pieces}")

        for p in pieces:
            tokens.append(vocab.get(p, vocab["<unk>"]))

    return tokens


def tokenize(text, vocab_path, merges_path):
    vocab = load_vocab(vocab_path)
    merges = load_merges(merges_path)

    text = "Hello world! This is a test of the BPE tokenizer. The quick brown fox jumps over the lazy dog."
    tokens = encode(text, vocab, merges)

    print(tokens)