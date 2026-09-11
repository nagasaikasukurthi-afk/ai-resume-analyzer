import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "resume_ner_v4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

id2label = model.config.id2label


# ============================================================
# PREDICT ENTITIES
# ============================================================

def predict_entities(text: str):

    if not text or not text.strip():
        return []

    # IMPORTANT:
    # V4 was trained using whitespace-separated words.
    words = text.split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()

    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**model_inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=-1
    )[0].cpu().tolist()


    # ========================================================
    # ONE PREDICTION PER ORIGINAL WORD
    # ========================================================

    word_predictions = {}

    for token_index, word_id in enumerate(word_ids):

        if word_id is None:
            continue

        # Keep ONLY the first subword prediction.
        if word_id not in word_predictions:

            word_predictions[word_id] = id2label[
                predictions[token_index]
            ]


    # ========================================================
    # RECONSTRUCT ENTITIES
    # ========================================================

    entities = []

    current_words = []
    current_label = None

    for word_index, word in enumerate(words):

        label = word_predictions.get(
            word_index,
            "O"
        )


        # ----------------------------------------------------
        # Outside
        # ----------------------------------------------------

        if label == "O":

            if current_words:

                entities.append({
                    "text": " ".join(current_words),
                    "label": current_label
                })

                current_words = []
                current_label = None

            continue


        # ----------------------------------------------------
        # Beginning
        # ----------------------------------------------------

        if label.startswith("B-"):

            if current_words:

                entities.append({
                    "text": " ".join(current_words),
                    "label": current_label
                })

            current_words = [word]
            current_label = label[2:]

            continue


        # ----------------------------------------------------
        # Inside
        # ----------------------------------------------------

        if label.startswith("I-"):

            entity_label = label[2:]

            if (
                current_words
                and current_label == entity_label
            ):

                current_words.append(word)

            else:

                if current_words:

                    entities.append({
                        "text": " ".join(current_words),
                        "label": current_label
                    })

                current_words = [word]
                current_label = entity_label


    # ========================================================
    # FINAL ENTITY
    # ========================================================

    if current_words:

        entities.append({
            "text": " ".join(current_words),
            "label": current_label
        })


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    cleaned_entities = []

    seen = set()

    for entity in entities:

        text_value = entity["text"].strip()

        if not text_value:
            continue

        key = (
            entity["label"],
            text_value.lower()
        )

        if key in seen:
            continue

        seen.add(key)

        cleaned_entities.append({
            "text": text_value,
            "label": entity["label"]
        })


    return cleaned_entities