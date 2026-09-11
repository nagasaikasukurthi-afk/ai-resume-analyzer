import json
from pathlib import Path
from collections import Counter

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

from weighted_trainer import WeightedTrainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = Path(
    "data/processed/ner_clean/train.json"
)

VALIDATION_FILE = Path(
    "data/processed/ner_clean/validation.json"
)

TEST_FILE = Path(
    "data/processed/ner_clean/test.json"
)

OUTPUT_DIR = Path(
    "models/resume_ner_v4"
)

MAX_LENGTH = 512


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NER V4 TRAINING")
print("=" * 70)

print("\nLoading datasets...")


with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as file:
    train_data = json.load(file)


with open(
    VALIDATION_FILE,
    "r",
    encoding="utf-8"
) as file:
    validation_data = json.load(file)


with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as file:
    test_data = json.load(file)


print(
    f"Train resumes:      {len(train_data)}"
)

print(
    f"Validation resumes: {len(validation_data)}"
)

print(
    f"Test resumes:       {len(test_data)}"
)


# ============================================================
# LABELS
# ============================================================

labels = sorted(
    {
        label
        for record in train_data
        for label in record["ner_tags"]
    }
)


label2id = {
    label: index
    for index, label in enumerate(labels)
}

id2label = {
    index: label
    for label, index in label2id.items()
}


print("\nLabels:")

for label, index in label2id.items():

    print(
        f"{index:2} -> {label}"
    )


num_labels = len(labels)

print(
    f"\nNumber of labels: {num_labels}"
)


# ============================================================
# CONVERT TO HUGGING FACE DATASET
# ============================================================

train_dataset = Dataset.from_list(
    train_data
)

validation_dataset = Dataset.from_list(
    validation_data
)

test_dataset = Dataset.from_list(
    test_data
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    all_labels = []

    for batch_index in range(
        len(examples["tokens"])
    ):

        word_ids = tokenized_inputs.word_ids(
            batch_index=batch_index
        )

        original_labels = examples["ner_tags"][
            batch_index
        ]

        labels_for_tokens = []

        previous_word_id = None

        for word_id in word_ids:

            if word_id is None:

                labels_for_tokens.append(-100)

            elif word_id != previous_word_id:

                labels_for_tokens.append(
                    label2id[
                        original_labels[word_id]
                    ]
                )

            else:

                # Ignore additional subword pieces.
                labels_for_tokens.append(-100)

            previous_word_id = word_id

        all_labels.append(
            labels_for_tokens
        )

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


print("Tokenizing datasets...")


tokenized_train = train_dataset.map(
    tokenize_and_align_labels,
    batched=True
)

tokenized_validation = validation_dataset.map(
    tokenize_and_align_labels,
    batched=True
)

tokenized_test = test_dataset.map(
    tokenize_and_align_labels,
    batched=True
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating moderate class weights...")


label_counter = Counter()

for record in train_data:

    for label in record["ner_tags"]:

        label_counter[label] += 1


total_labels = sum(
    label_counter.values()
)


weights = []


for label in labels:

    count = label_counter[label]

    # Square-root inverse frequency.
    weight = (
        total_labels / count
    ) ** 0.5

    weights.append(weight)


weights = torch.tensor(
    weights,
    dtype=torch.float
)


# Normalize weights around 1.
weights = weights / weights.mean()


# Keep weights within a reasonable range.
weights = torch.clamp(
    weights,
    min=0.25,
    max=5.0
)


print("\nClass weights:")

for label, weight in zip(
    labels,
    weights
):

    print(
        f"{label:30} "
        f"count={label_counter[label]:6} "
        f"weight={weight.item():.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print("\nLoading model...")


model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(OUTPUT_DIR),

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=8,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=2,

    report_to="none"
)


# ============================================================
# METRICS
# ============================================================

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def compute_metrics(eval_prediction):

    predictions, labels_array = eval_prediction

    predictions = predictions.argmax(
        axis=-1
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels_array
    ):

        current_predictions = []
        current_labels = []

        for predicted_id, true_id in zip(
            prediction,
            label
        ):

            if true_id == -100:
                continue

            current_predictions.append(
                id2label[int(predicted_id)]
            )

            current_labels.append(
                id2label[int(true_id)]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    return {
        "precision": precision_score(
            true_labels,
            true_predictions
        ),

        "recall": recall_score(
            true_labels,
            true_predictions
        ),

        "f1": f1_score(
            true_labels,
            true_predictions
        ),

        "accuracy": accuracy_score(
            true_labels,
            true_predictions
        )
    }


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    class_weights=weights,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING V4 TRAINING")
print("=" * 70)


trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving V4 model...")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)


validation_results = trainer.evaluate(
    eval_dataset=tokenized_validation
)


for key, value in validation_results.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)


test_results = trainer.evaluate(
    eval_dataset=tokenized_test
)


for key, value in test_results.items():

    print(
        f"{key}: {value}"
    )


print("\n" + "=" * 70)
print("NER V4 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nModel saved to: {OUTPUT_DIR}"
)