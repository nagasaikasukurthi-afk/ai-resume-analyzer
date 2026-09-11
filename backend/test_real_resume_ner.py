from app.services.resume_parser import extract_resume_text
from app.services.ner_extractor import predict_entities


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "uploads" / "Naga Sai.pdf"


print("=" * 70)
print("REAL RESUME - NER V4 TEST")
print("=" * 70)


# ============================================================
# EXTRACT RESUME TEXT
# ============================================================

resume_text = extract_resume_text(PDF_PATH)

print(f"\nExtracted characters: {len(resume_text)}")


# ============================================================
# RUN NER V4
# ============================================================

entities = predict_entities(resume_text)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("NER V4 PREDICTED ENTITIES")
print("=" * 70)


if not entities:

    print("No entities detected.")

else:

    for entity in entities:

        print(
            f"{entity['label']:25} -> {entity['text']}"
        )


# ============================================================
# SUMMARY BY ENTITY TYPE
# ============================================================

print("\n")
print("=" * 70)
print("ENTITY SUMMARY")
print("=" * 70)


entity_counts = {}

for entity in entities:

    label = entity["label"]

    entity_counts[label] = (
        entity_counts.get(label, 0) + 1
    )


for label, count in sorted(
    entity_counts.items()
):

    print(
        f"{label:25} -> {count}"
    )


print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)