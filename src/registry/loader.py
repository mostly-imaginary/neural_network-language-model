import os
import json
from src.tokenizer.bpe_tokenize import *

# Model loading and training logic, main thing to run to train the model and generate merges/vocab

def load_registry(path="models/registry.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["models"]
    
def find_model_by_name(models, name):
    for model in models:
        if model["name"] == name:
            return model
    return None

def load_config(model_path):
    config_path = f"{model_path}/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(name):
    models = load_registry()

    model_info = find_model_by_name(models, name)

    if model_info is None:
        raise ValueError(f"Model '{name}' not found")

    config = load_config(model_info["path"])

    return config