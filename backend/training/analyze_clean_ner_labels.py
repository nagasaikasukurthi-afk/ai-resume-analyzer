import json
from pathlib import Path
from collections import Counter


DATA_DIR = Path("data/processed/ner_clean")


FILES = {
    "TRAIN": DATA_DIR / "train.json",
    "VALIDATION": DATA_DIR / "validation.json",
    "TEST": DATA_DIR / "test.json",
}


print("=" * 70)
print("CLEAN NER LABEL DISTRIBUTION")
print("=" * 70)


for split_name, file_path in FILES.items():

    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    counter = Counter()

    for record in dataset:

        counter.update(
            record["ner_tags"]
        )

    total_tokens = sum(counter.values())

    print(f"Resumes: {len(dataset)}")
    print(f"Total tokens: {total_tokens}")

    print("\nLabel distribution:")

    for label, count in counter.most_common():

        percentage = (
            count / total_tokens
        ) * 100

        print(
            f"{label:30} "
            f"{count:8} "
            f"{percentage:7.2f}%"
        )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
