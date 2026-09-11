from datasets import load_dataset


DATASET_NAME = "yashpwr/resume-ner-training-data"


print("Loading resume NER dataset...")

dataset = load_dataset(DATASET_NAME)

print("\n===== DATASET =====")
print(dataset)

print("\n===== SPLITS =====")
print(dataset.keys())

for split in dataset.keys():
    print(f"\n===== {split.upper()} =====")
    print("Number of examples:", len(dataset[split]))
    print("Columns:", dataset[split].column_names)

    print("\nFirst example:")
    print(dataset[split][0])