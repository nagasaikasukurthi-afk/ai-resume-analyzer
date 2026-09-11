import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/processed/ner")


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


for split in ["train", "validation", "test"]:

    data = load_dataset(
        DATA_DIR / f"{split}.json"
    )

    counter = Counter()

    for record in data:
        counter.update(record["ner_tags"])

    total = sum(counter.values())

    print("\n" + "=" * 60)
    print(f"{split.upper()} LABEL DISTRIBUTION")
    print("=" * 60)

    for label, count in counter.most_common():

        percentage = (
            count / total
        ) * 100

        print(
            f"{label:30} "
            f"{count:6} "
            f"{percentage:6.2f}%"
        )

    print(f"\nTotal tokens: {total}")