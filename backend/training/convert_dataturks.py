import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/ner/traindata.json")
OUTPUT_FILE = Path("data/processed/ner_dataset.json")


def load_dataturks_dataset():
    data = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                data.append(record)

            except json.JSONDecodeError as e:
                print(
                    f"Skipping invalid line {line_number}: {e}"
                )

    return data


def tokenize_with_offsets(text):
    """
    Tokenize text using whitespace while preserving
    exact character offsets.
    """

    tokens = []

    for match in re.finditer(r"\S+", text):

        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return tokens


def get_valid_annotations(annotations):
    """
    Extract valid DataTurks annotations.
    """

    valid_annotations = []

    for annotation in annotations:

        labels = annotation.get("label", [])

        if not labels:
            continue

        label = labels[0]

        # Ignore unknown labels
        if label == "UNKNOWN":
            continue

        for point in annotation.get("points", []):

            start = point.get("start")
            end = point.get("end")

            if start is None or end is None:
                continue

            if end <= start:
                continue

            valid_annotations.append({
                "label": label,
                "start": start,
                "end": end
            })

    return valid_annotations


def token_overlaps_entity(token, entity):

    return (
        token["start"] < entity["end"]
        and token["end"] > entity["start"]
    )


def convert_record(item, record_id):

    text = item.get("content", "")

    annotations = get_valid_annotations(
        item.get("annotation", [])
    )

    tokens = tokenize_with_offsets(text)

    token_texts = []
    ner_tags = []

    # Keep track of which entity each token belongs to
    previous_entity = None

    for token in tokens:

        token_texts.append(
            token["text"]
        )

        matching_entities = [
            entity
            for entity in annotations
            if token_overlaps_entity(
                token,
                entity
            )
        ]

        if not matching_entities:

            ner_tags.append("O")
            previous_entity = None
            continue

        # If overlapping annotations exist,
        # choose the entity with the largest overlap.
        entity = max(
            matching_entities,
            key=lambda e:
                min(token["end"], e["end"])
                - max(token["start"], e["start"])
        )

        entity_key = (
            entity["label"],
            entity["start"],
            entity["end"]
        )

        if previous_entity == entity_key:

            ner_tags.append(
                f"I-{entity['label']}"
            )

        else:

            ner_tags.append(
                f"B-{entity['label']}"
            )

        previous_entity = entity_key

    return {
        "id": record_id,
        "tokens": token_texts,
        "ner_tags": ner_tags
    }


def main():

    print("Loading DataTurks dataset...")

    data = load_dataturks_dataset()

    print(
        f"Total resumes loaded: {len(data)}"
    )

    converted_data = []

    for index, item in enumerate(data):

        converted_data.append(
            convert_record(
                item,
                index
            )
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            converted_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n===== CONVERSION COMPLETE =====")

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Records: {len(converted_data)}"
    )

    print("\n===== FIRST RECORD =====")

    first = converted_data[0]

    for token, label in zip(
        first["tokens"][:60],
        first["ner_tags"][:60]
    ):

        print(
            f"{token:30} {label}"
        )


if __name__ == "__main__":
    main()