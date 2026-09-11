import json
import re
from pathlib import Path
from collections import Counter


RAW_FILE = Path("data/raw/ner/traindata.json")
OUTPUT_FILE = Path("data/processed/ner_clean_dataset.json")


VALID_LABELS = {
    "Name",
    "College Name",
    "Degree",
    "Graduation Year",
    "Years of Experience",
    "Companies worked at",
    "Designation",
    "Skills",
    "Location",
    "Email Address",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                print(
                    f"Warning: Could not parse line {line_number}: {error}"
                )

    return records


def get_valid_entities(record):

    text = record.get("content", "")

    annotations = record.get("annotation", [])

    entities = []

    for annotation in annotations:

        label = annotation.get("label")

        # Some DataTurks records contain empty label lists.
        if isinstance(label, list):

            if not label:
                continue

            label = label[0]

        # Ignore missing/unknown labels.
        if not label:
            continue

        if label not in VALID_LABELS:
            continue

        points = annotation.get("points", [])

        for point in points:

            start = point.get("start")
            end = point.get("end")

            if start is None or end is None:
                continue

            # Invalid character positions.
            if start < 0 or end < start:
                continue

            if start >= len(text):
                continue

            # DataTurks end position is inclusive.
            actual_end = min(end + 1, len(text))

            actual_text = text[start:actual_end]

            # IMPORTANT:
            # Keep only annotations whose stored text
            # exactly matches the source text.
            expected_text = point.get("text", "")

            if actual_text != expected_text:
                continue

            if not actual_text.strip():
                continue

            entities.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": actual_text,
                }
            )

    return entities


def tokenize_with_offsets(text):

    tokens = []

    for match in re.finditer(r"\S+", text):

        tokens.append(
            {
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
            }
        )

    return tokens


def convert_to_bio(text, entities):

    tokens = tokenize_with_offsets(text)

    ner_tags = ["O"] * len(tokens)

    # Sort entities by start position.
    entities = sorted(
        entities,
        key=lambda entity: (
            entity["start"],
            entity["end"]
        )
    )

    previous_entity = None

    for entity in entities:

        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_label = entity["label"]

        entity_token_indexes = []

        for token_index, token in enumerate(tokens):

            token_start = token["start"]
            token_end = token["end"]

            # Check character overlap.
            overlaps = (
                token_start < entity_end + 1
                and token_end > entity_start
            )

            if overlaps:
                entity_token_indexes.append(token_index)

        if not entity_token_indexes:
            continue

        for position, token_index in enumerate(entity_token_indexes):

            prefix = "B" if position == 0 else "I"

            ner_tags[token_index] = (
                f"{prefix}-{entity_label}"
            )

        previous_entity = entity

    return (
        [token["text"] for token in tokens],
        ner_tags
    )


print("=" * 70)
print("CLEAN DATATURKS → BIO CONVERSION")
print("=" * 70)


print("\nLoading original DataTurks dataset...")

records = load_jsonl(RAW_FILE)

print(f"Original resumes: {len(records)}")


clean_dataset = []

total_annotations = 0
valid_annotations = 0
discarded_annotations = 0

discard_reasons = Counter()
label_counts = Counter()


for index, record in enumerate(records):

    text = record.get("content", "")

    annotations = record.get("annotation", [])

    total_annotations += len(annotations)

    valid_entities = get_valid_entities(record)

    valid_annotations += len(valid_entities)

    discarded_annotations += (
        len(annotations) - len(valid_entities)
    )

    # Count discarded annotations approximately by
    # checking every original annotation.
    for annotation in annotations:

        label = annotation.get("label")

        if isinstance(label, list):

            if not label:
                discard_reasons["empty_label"] += 1
                continue

            label = label[0]

        if not label:
            discard_reasons["missing_label"] += 1

        elif label not in VALID_LABELS:
            discard_reasons["unknown_or_invalid_label"] += 1

        else:

            points = annotation.get("points", [])

            annotation_valid = False

            for point in points:

                start = point.get("start")
                end = point.get("end")

                if start is None or end is None:
                    continue

                if start < 0 or end < start:
                    continue

                if start >= len(text):
                    continue

                actual_end = min(
                    end + 1,
                    len(text)
                )

                actual_text = text[start:actual_end]

                expected_text = point.get(
                    "text",
                    ""
                )

                if (
                    actual_text == expected_text
                    and actual_text.strip()
                ):
                    annotation_valid = True
                    break

            if not annotation_valid:
                discard_reasons[
                    "span_text_mismatch_or_invalid_span"
                ] += 1

    for entity in valid_entities:
        label_counts[entity["label"]] += 1

    tokens, ner_tags = convert_to_bio(
        text,
        valid_entities
    )

    clean_dataset.append(
        {
            "id": index,
            "tokens": tokens,
            "ner_tags": ner_tags,
        }
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
        clean_dataset,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(
    f"Total original annotations: "
    f"{total_annotations}"
)

print(
    f"Valid annotations kept: "
    f"{valid_annotations}"
)

print(
    f"Annotations discarded: "
    f"{discarded_annotations}"
)


print("\nDiscard reasons:")

for reason, count in discard_reasons.items():

    print(
        f"  {reason}: {count}"
    )


print("\nValid entity counts:")

for label, count in sorted(
    label_counts.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        f"  {label:25} {count}"
    )


print("\n" + "=" * 70)
print("DATASET VALIDATION")
print("=" * 70)


print(
    f"Clean resumes created: "
    f"{len(clean_dataset)}"
)


all_lengths_match = True

for record in clean_dataset:

    if len(record["tokens"]) != len(record["ner_tags"]):

        all_lengths_match = False

        print(
            f"Length mismatch in resume "
            f"{record['id']}"
        )


print(
    f"Token/label lengths match: "
    f"{all_lengths_match}"
)


print("\nOutput file:")

print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("CLEAN CONVERSION COMPLETE")
print("=" * 70)