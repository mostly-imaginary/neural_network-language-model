# Neural Network Language Model

An experimental project focused on building a lightweight language model with reduced computational requirements.

The project currently implements:
- Byte Pair Encoding (BPE) tokenizer (fully functional)
- Config-driven model system (in progress)
- Registry-based model loading
- Dataset pipeline (Wikipedia subset)

---

# Setup

## 1. Create virtual environment

### Windows:
python -m venv .venv

### Mac/Linux:
source .venv/bin/activate

## 2. Install dependencies
pip install -r requirements.txt

# Project Structure
neural_network-language-model/
│
├── datasets/
│   └── wikipedia/
│       └── v1/
│
├── download/
│   └── fetch_wiki_dataset.py
│
├── models/
│   ├── registry.json
│   └── chatnotgpt/
│       └── v1/
│           ├── config.json
│
├── src/
│   ├── model/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── __init__.py
│   │
│   ├── registry/
│   │   ├── loader.py
│   │   └── __init__.py
│   │
│   ├── tokenizer/
│   │   ├── bpe_tokenize.py
│   │   ├── bpe_train.py
│   │   └── __init__.py
│   │
│   └── utils/
│
└── tests/

# Dataset setup

## 1. Create dataset folder
Make sure datasets/wikipedia/v1 exists

## 2. Download dataset
Run: python -m download.fetch_wiki_dataset
This generates: datasets/wikipedia/v1/wiki_subset.txt

# Training (BPE only for now)
Run: python -m src.model.train

This will:

1. Load dataset
2. Build character-level corpus
3. Train BPE merges
4. Save tokenizer artifacts:
* bpe_merges.txt
* bpe_vocab.txt

# Model System
Models are defined using a registry system:

## Registry file
models/registry.json

### Example:
{
  "models": [
    {
      "id": "chatnotgpt_v1",
      "name": "ChatNotGPT V1",
      "path": "models/chatnotgpt/v1",
      "version": "v1",
      "tokenizer": "BPE"
    }
  ]
}

## Configuration file
models/path_to_model/config.json

### Example config.json:
{
  "model_id": "chatnotgpt_v1",
  "model_name": "ChatNotGPT",
  "version": "v1",

  "tokenizer": {
    "type": "BPE",
    "vocab_size": 2001,
    "merges_path": "bpe_merges.txt",
    "vocab_path": "bpe_vocab.txt",
    "special_tokens": ["<unk>"],
    "dataset_path": "datasets/wikipedia/v1/wiki_subset.txt"
  },

  "model": {
    "embedding_dim": 128,
    "context_size": 100
  },

  "training": {
    "batch_size": 32,
    "learning_rate": 0.001
  },

  "layers": [
    {
      "type": "linear",
      "activation": "relu",
      "size": 128
    },
    {
      "type": "linear",
      "activation": "relu",
      "size": 128
    },
    {
      "type": "linear",
      "activation": "softmax",
      "size": 1000
    }
  ]
}

# Tokenization details
* Tokenizer is character-level BPE
* End-of-word marker: \uE000
* <unk> token is used for unknown merges or unseen tokens
* Output tokens are integer IDs mapped from bpe_vocab.txt

# Current limitations
* No neural network yet
* Tokenizer is the only working component

# Future improvements
* Adding Neural Network
* Adding embeddings (Hopefully soon)
* Faster BPE