import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path("models/resume_ner_v3")

TEST_FILE = Path(
    "data/processed/ner/test.json"
)


# ============================================================
# Load test dataset
# ============================================================

print("Loading test dataset...")

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:

    test_data = json.load(file)


print(
    f"Test resumes: {len(test_data)}"
)


# ============================================================
# Load model
# ============================================================

print("\nLoading trained NER model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("Model loaded.")


# ============================================================
# Evaluate
# ============================================================

true_labels_all = []
predicted_labels_all = []


for index, example in enumerate(test_data):

    tokens = example["tokens"]

    true_word_labels = example["ner_tags"]

    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    word_ids = encoded.word_ids(
        batch_index=0
    )

    outputs = model(
        **encoded
    )

    predictions = np.argmax(
        outputs.logits.detach().numpy(),
        axis=2
    )[0]

    true_labels = []
    predicted_labels = []

    previous_word_id = None

    for token_index, word_id in enumerate(
        word_ids
    ):

        if word_id is None:
            continue

        # Only evaluate the first sub-token
        if word_id == previous_word_id:
            continue

        if word_id >= len(true_word_labels):
            continue

        true_labels.append(
            true_word_labels[word_id]
        )

        predicted_labels.append(
            model.config.id2label[
                int(predictions[token_index])
            ]
        )

        previous_word_id = word_id

    true_labels_all.append(
        true_labels
    )

    predicted_labels_all.append(
        predicted_labels
    )


# ============================================================
# Overall metrics
# ============================================================

precision = precision_score(
    true_labels_all,
    predicted_labels_all
)

recall = recall_score(
    true_labels_all,
    predicted_labels_all
)

f1 = f1_score(
    true_labels_all,
    predicted_labels_all
)


print("\n")
print("=" * 70)
print("OVERALL NER RESULTS")
print("=" * 70)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)


# ============================================================
# Entity-level report
# ============================================================

print("\n")
print("=" * 70)
print("ENTITY-LEVEL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        true_labels_all,
        predicted_labels_all,
        digits=4
    )
)


# ============================================================
# Prediction examples
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)


for index in range(
    min(3, len(test_data))
):

    print(
        f"\n===== RESUME {index + 1} ====="
    )

    tokens = test_data[index]["tokens"]

    true_labels = true_labels_all[index]

    predicted_labels = predicted_labels_all[index]

    for token, true_label, predicted_label in zip(
        tokens,
        true_labels,
        predicted_labels
    ):

        if (
            true_label != "O"
            or predicted_label != "O"
        ):

            print(
                f"{token:30} "
                f"TRUE={true_label:25} "
                f"PRED={predicted_label}"
            )