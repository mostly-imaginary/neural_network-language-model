from src.registry.loader import load_model
from src.tokenizer import bpe_train

def train_model(model_name):
    config = load_model(model_name)

    bpe_train.train(
        vocab_size=config["tokenizer"]["vocab_size"],
        merges_path=config["tokenizer"]["merges_path"],
        vocab_path=config["tokenizer"]["vocab_path"],
        dataset_path=config["tokenizer"]["dataset_path"]
    )

if __name__ == "__main__":
    train_model("ChatNotGPT V1")