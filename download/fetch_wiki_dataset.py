import os
import re
from datasets import load_dataset

# This script fetches a subset of the Wikipedia dataset and saves it to a text file. Make sure the directory "datasets/wikipedia/v1/" exists before running this script.

def clean_text(text):
    text = text.replace("\u200e", "")
    text = text.replace("\u200f", "")

    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)

    text = " ".join(text.split())

    return text


def save_wiki_subset(output_path="datasets/wikipedia/v1/wiki_subset.txt", n=10000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{n}]"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"]

            text = clean_text(text)

            f.write(text + "\n")

    print(f"Saved {n} articles to:", output_path)


if __name__ == "__main__":
    save_wiki_subset()