from model_wrappers import GPT2DiffusionTransformer
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import json

from corruption_utils import (
    SimilarTokenCorruption,
    RandomTokenCorruption,
    MaskTokenCorruption,
    MixedTokenCorruption,
)

from train_utils import (
    DiffusionDataCollator,
    TrainingOutput,
    train
)

MODEL_NAME = "openai-community/gpt2"

DATASET_PATH = "Salesforce/wikitext"
DATASER_NAME = "wikitext-2-raw-v1"

ITERATIONS_INTERVALS = [5, 2, 2]
LEARNING_RATE = 1e-5

RESULT_FOLDER = "results"
MODEL_SAVE_PATH = f"{RESULT_FOLDER}/models"
MODEL_ID = "v1"
TRAINING_OUTPUT_SAVE_PATH = f"{RESULT_FOLDER}/training_output"
TRAIN_OUTPUT_ID = "v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "mask_token": "<|mask|>",
})

model = GPT2DiffusionTransformer.from_file_path(
    file_path=f"{MODEL_SAVE_PATH}/{MODEL_ID}.pt",
    model_name=MODEL_NAME,
    num_diffusion_steps=1000,
    vocabulary_size=len(tokenizer)
)

ds = load_dataset(DATASET_PATH, DATASER_NAME)

dataset_test = ds['test']
dataset_train = ds['train']

corruption_method_1 = SimilarTokenCorruption(
    embedding_weight=model.transformer.wte.weight,
    num_diffusion_steps=100,
    number_of_neighbors=20,
    minimum_probability=0.01,
    maximum_probability=0.30,
)

corruption_method_2 = RandomTokenCorruption(
    dictionary_size=len(tokenizer),
    num_diffusion_steps=100,
    minimum_probability=0.01,
    maximum_probability=0.95,
)

corruption_method_3 = MaskTokenCorruption(
    mask_token_id=tokenizer.mask_token_id,
    num_diffusion_steps=100,
    minimum_probability=0.01,
    maximum_probability=0.95,
)


corruption_method = MixedTokenCorruption(
    corruption_methods=[
        corruption_method_1,
        corruption_method_2,
        corruption_method_3,
    ],
    iterations_intervals=ITERATIONS_INTERVALS,
)

collator = DiffusionDataCollator(
    tokenizer=tokenizer,
    corruption_method=corruption_method,
    num_diffusion_steps=100,
    max_length=64,
)

train_loader = DataLoader(
    dataset_train,
    batch_size=95,
    shuffle=True,
    collate_fn=collator,
)

test_loader = DataLoader(
    dataset_test,
    batch_size=95,
    shuffle=True,
    collate_fn=collator,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

training_output: TrainingOutput = train(device, model, optimizer, sum(ITERATIONS_INTERVALS), train_loader, test_loader)

Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAINING_OUTPUT_SAVE_PATH).mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/{MODEL_ID}.pt")
with open(f"{MODEL_SAVE_PATH}/{MODEL_ID}.json", "w") as f:
    json.dump(training_output.to_dict(),f, indent=4)