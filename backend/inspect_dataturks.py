import json

DATASET_PATH = "data/raw/ner/traindata.json"

print("Loading DataTurks resume NER dataset...")

data = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
            data.append(record)
        except json.JSONDecodeError as e:
            print(f"Error on line {line_number}: {e}")

print("\n===== DATASET =====")
print("Number of resumes:", len(data))

print("\n===== FIRST RECORD =====")

first_item = data[0]

print("\nKeys:")
print(first_item.keys())

print("\n===== RESUME TEXT =====")
print(first_item["content"][:1500])

print("\n===== ANNOTATIONS =====")

for annotation in first_item.get("annotation", []):
    print(annotation)

print("\n===== ENTITY LABELS =====")

labels = set()

for item in data:
    for annotation in item.get("annotation", []):
        for label in annotation.get("label", []):
            labels.add(label)

print("\n".join(sorted(labels)))

print("\n===== ANNOTATION COUNTS =====")

annotation_counts = {}

for item in data:
    for annotation in item.get("annotation", []):
        for label in annotation.get("label", []):
            annotation_counts[label] = annotation_counts.get(label, 0) + 1

for label, count in sorted(annotation_counts.items()):
    print(f"{label}: {count}")