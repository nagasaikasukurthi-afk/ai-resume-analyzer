import json
from pathlib import Path


RAW_FILE = Path("data/raw/ner/traindata.json")
PROCESSED_FILE = Path("data/processed/ner_dataset.json")


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def get_original_entities(record):
    entities = []

    annotations = record.get("annotation", [])

    for annotation in annotations:

        label = annotation.get("label")

        if isinstance(label, list):

            if not label:
                continue

            label = label[0]

        if not label:
             continue


        if label == "UNKNOWN":
            continue

        points = annotation.get("points", [])

        for point in points:

            entities.append({
                "label": label,
                "start": point["start"],
                "end": point["end"],
                "text": point["text"]
            })

    return entities


print("=" * 70)
print("NER DATASET CONVERSION AUDIT")
print("=" * 70)


print("\nLoading original DataTurks data...")

raw_records = load_jsonl(RAW_FILE)

print(f"Original resumes: {len(raw_records)}")


print("\nLoading processed BIO dataset...")

with open(PROCESSED_FILE, "r", encoding="utf-8") as file:
    processed_records = json.load(file)

print(f"Processed resumes: {len(processed_records)}")


print("\n" + "=" * 70)
print("CHECKING ENTITY ANNOTATIONS")
print("=" * 70)


issues = []

for index, record in enumerate(raw_records):

    text = record["content"]

    entities = get_original_entities(record)

    for entity in entities:

        start = entity["start"]
        end = entity["end"]
        entity_text = entity["text"]

        actual_text = text[start:end + 1]

        if actual_text != entity_text:

            issues.append({
                "resume": index,
                "label": entity["label"],
                "expected": entity_text,
                "actual": actual_text,
                "start": start,
                "end": end
            })


print(f"\nAnnotation text mismatches: {len(issues)}")


if issues:

    print("\nFirst 20 mismatches:")

    for issue in issues[:20]:

        print("\nResume:", issue["resume"])
        print("Label:", issue["label"])
        print("Expected:", repr(issue["expected"]))
        print("Actual:", repr(issue["actual"]))
        print("Span:", issue["start"], "-", issue["end"])


print("\n" + "=" * 70)
print("ENTITY SAMPLE INSPECTION")
print("=" * 70)


for index in range(min(10, len(raw_records))):

    record = raw_records[index]

    print(f"\n===== RESUME {index + 1} =====")

    entities = get_original_entities(record)

    for entity in entities[:15]:

        print(
            f"{entity['label']:25} "
            f"{repr(entity['text'])}"
        )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)