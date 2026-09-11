import json
import random
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/ner_clean_dataset.json"
)

OUTPUT_DIR = Path(
    "data/processed/ner_clean"
)

SEED = 42


print("=" * 70)
print("SPLITTING CLEAN NER DATASET")
print("=" * 70)


print("\nLoading clean dataset...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:
    dataset = json.load(file)


print(f"Total resumes: {len(dataset)}")


# Create a copy so the original clean dataset
# is never modified.
dataset = list(dataset)


# Reproducible shuffle.
random.seed(SEED)
random.shuffle(dataset)


total = len(dataset)

train_size = int(total * 0.80)
validation_size = int(total * 0.10)


train_data = dataset[:train_size]

validation_data = dataset[
    train_size:
    train_size + validation_size
]

test_data = dataset[
    train_size + validation_size:
]


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_json(data, filename):

    output_file = OUTPUT_DIR / filename

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved {filename}: "
        f"{len(data)} resumes"
    )


print("\nCreating splits...")

save_json(
    train_data,
    "train.json"
)

save_json(
    validation_data,
    "validation.json"
)

save_json(
    test_data,
    "test.json"
)


print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(f"Total:      {len(dataset)}")
print(f"Training:   {len(train_data)}")
print(f"Validation: {len(validation_data)}")
print(f"Test:       {len(test_data)}")


print("\nOutput directory:")
print(OUTPUT_DIR)


print("\n" + "=" * 70)
print("CLEAN NER SPLIT COMPLETE")
print("=" * 70)