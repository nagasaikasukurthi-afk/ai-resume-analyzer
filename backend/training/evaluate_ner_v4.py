import json
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resume_ner_v4"
TEST_DATA_PATH = "data/processed/ner_clean/test.json"


# ============================================================
# LOAD LABELS
# ============================================================

with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

label_names = sorted(
    {
        label
        for item in test_data
        for label in item["ner_tags"]
    }
)

label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

print("=" * 70)
print("NER V4 ENTITY-LEVEL EVALUATION")
print("=" * 70)

print("\nLabels:")
for label in label_names:
    print(f"{label2id[label]:2d}: {label}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print(f"Device: {device}")


# ============================================================
# TOKENIZE TEST DATA
# ============================================================

test_dataset = Dataset.from_list(test_data)


def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )

    word_ids = tokenized_inputs.word_ids()

    labels = []

    previous_word_id = None

    for word_id in word_ids:

        if word_id is None:
            labels.append(-100)

        elif word_id != previous_word_id:
            labels.append(
                label2id[example["ner_tags"][word_id]]
            )

        else:
            labels.append(-100)

        previous_word_id = word_id

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


tokenized_test = test_dataset.map(
    tokenize_and_align_labels
)


data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

print("\nRunning predictions...")

for example in tokenized_test:

    input_ids = torch.tensor(
        [example["input_ids"]]
    ).to(device)

    attention_mask = torch.tensor(
        [example["attention_mask"]]
    ).to(device)

    labels = example["labels"]

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )[0].cpu().numpy()

    prediction_labels = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        if label == -100:
            continue

        prediction_labels.append(
            id2label[int(prediction)]
        )

        true_labels.append(
            id2label[int(label)]
        )

    all_predictions.append(prediction_labels)
    all_labels.append(true_labels)


# ============================================================
# OVERALL RESULTS
# ============================================================

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n")
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")


# ============================================================
# ENTITY-LEVEL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ENTITY-LEVEL RESULTS")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)


def extract_entities(tokens, labels):

    entities = []

    current_entity = []
    current_label = None

    for token, label in zip(tokens, labels):

        if label.startswith("B-"):

            if current_entity:
                entities.append(
                    (
                        " ".join(current_entity),
                        current_label
                    )
                )

            current_entity = [token]
            current_label = label[2:]

        elif label.startswith("I-"):

            entity_label = label[2:]

            if (
                current_entity
                and current_label == entity_label
            ):
                current_entity.append(token)

            else:

                if current_entity:
                    entities.append(
                        (
                            " ".join(current_entity),
                            current_label
                        )
                    )

                current_entity = [token]
                current_label = entity_label

        else:

            if current_entity:
                entities.append(
                    (
                        " ".join(current_entity),
                        current_label
                    )
                )

                current_entity = []
                current_label = None

    if current_entity:
        entities.append(
            (
                " ".join(current_entity),
                current_label
            )
        )

    return entities


for index in range(min(5, len(test_data))):

    original = test_data[index]

    print("\n" + "-" * 70)
    print(f"RESUME {index + 1}")
    print("-" * 70)

    tokens = original["tokens"]

    actual_labels = original["ner_tags"]

    predicted_labels = []

    # Find corresponding prediction
    predicted_labels = all_predictions[index]

    # Display actual entities
    actual_entities = extract_entities(
        tokens[:len(actual_labels)],
        actual_labels
    )

    predicted_entities = extract_entities(
        tokens[:len(predicted_labels)],
        predicted_labels
    )

    print("\nACTUAL ENTITIES:")

    for entity, label in actual_entities[:20]:
        print(f"  {label:25} -> {entity}")

    print("\nPREDICTED ENTITIES:")

    for entity, label in predicted_entities[:20]:
        print(f"  {label:25} -> {entity}")


print("\n")
print("=" * 70)
print("V4 EVALUATION COMPLETE")
print("=" * 70)