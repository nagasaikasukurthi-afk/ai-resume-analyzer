import re

from app.services.resume_analyzer import analyze_resume
from app.services.ner_extractor import predict_entities


# ============================================================
# NER ENTITIES WE WANT TO CONSIDER
# ============================================================

ALLOWED_NER_ENTITIES = {
    "Name",
    "Location",
    "Designation",
    "Companies worked at",
    "College Name",
    "Degree",
}


# ============================================================
# GENERIC / INVALID VALUES
# ============================================================

INVALID_ENTITY_WORDS = {
    "btech",
    "b.tech",
    "mtech",
    "m.tech",
    "computer",
    "engineering",
    "python",
    "java",
    "javascript",
    "sql",
    "django",
    "fastapi",
    "postgresql",
    "mysql",
    "react",
    "github",
    "email",
    "resume",
    "curriculum vitae",
}


# ============================================================
# BASIC ENTITY VALIDATION
# ============================================================

def is_valid_entity(text: str, label: str) -> bool:

    if not text:
        return False

    text = text.strip()

    # Too short
    if len(text) < 3:
        return False

    # Ignore obvious email predictions for non-email entities
    if "@" in text and label != "Email Address":
        return False

    # Ignore URLs
    if text.lower().startswith(("http://", "https://", "www.")):
        return False

    normalized = text.lower().strip()

    # Ignore obvious generic predictions
    if normalized in INVALID_ENTITY_WORDS:
        return False

    # Ignore very short alphabetic fragments
    if len(normalized.split()) == 1 and len(normalized) <= 3:
        return False

    return True


# ============================================================
# CLEAN NER OUTPUT
# ============================================================

def clean_ner_entities(entities):

    cleaned = []

    seen = set()

    for entity in entities:

        label = entity.get("label")
        text = entity.get("text", "").strip()

        if label not in ALLOWED_NER_ENTITIES:
            continue

        if not is_valid_entity(text, label):
            continue

        key = (
            label,
            text.lower()
        )

        if key in seen:
            continue

        seen.add(key)

        cleaned.append({
            "text": text,
            "label": label
        })

    return cleaned


# ============================================================
# GROUP ENTITIES
# ============================================================

def group_entities(entities):

    grouped = {
        "names": [],
        "locations": [],
        "designations": [],
        "companies": [],
        "colleges": [],
        "degrees": [],
    }

    for entity in entities:

        label = entity["label"]
        text = entity["text"]

        if label == "Name":
            grouped["names"].append(text)

        elif label == "Location":
            grouped["locations"].append(text)

        elif label == "Designation":
            grouped["designations"].append(text)

        elif label == "Companies worked at":
            grouped["companies"].append(text)

        elif label == "College Name":
            grouped["colleges"].append(text)

        elif label == "Degree":
            grouped["degrees"].append(text)

    return grouped


# ============================================================
# FIND BEST NAME
# ============================================================

def select_name(entities, deterministic_result):

    # NER candidates
    name_candidates = [
        entity["text"]
        for entity in entities
        if entity["label"] == "Name"
    ]

    # Remove candidates that look like emails
    name_candidates = [
        name
        for name in name_candidates
        if "@" not in name
    ]

    if name_candidates:
        return name_candidates[0]

    return None


# ============================================================
# FIND LOCATIONS
# ============================================================

def select_locations(entities):

    locations = [
        entity["text"]
        for entity in entities
        if entity["label"] == "Location"
    ]

    return locations


# ============================================================
# HYBRID ANALYZER
# ============================================================

def analyze_resume_hybrid(text: str):

    if not text or not text.strip():

        return {
            "name": None,
            "email": None,
            "phone": None,
            "skills": [],
            "locations": [],
            "designations": [],
            "companies": [],
            "colleges": [],
            "degrees": [],
            "education": None,
            "experience": None,
            "projects": None,
            "certifications": None,
        }


    # ========================================================
    # RULE-BASED ANALYSIS
    # ========================================================

    deterministic = analyze_resume(text)


    # ========================================================
    # NER V4
    # ========================================================

    ner_entities = predict_entities(text)

    ner_entities = clean_ner_entities(
        ner_entities
    )


    # ========================================================
    # NER FIELDS
    # ========================================================

    name = select_name(
        ner_entities,
        deterministic
    )

    locations = select_locations(
        ner_entities
    )

    designations = [
        entity["text"]
        for entity in ner_entities
        if entity["label"] == "Designation"
    ]

    companies = [
        entity["text"]
        for entity in ner_entities
        if entity["label"] == "Companies worked at"
    ]

    colleges = [
        entity["text"]
        for entity in ner_entities
        if entity["label"] == "College Name"
    ]

    degrees = [
        entity["text"]
        for entity in ner_entities
        if entity["label"] == "Degree"
    ]


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        # NER-enhanced identity
        "name": name,

        # Reliable deterministic fields
        "email": deterministic["email"],
        "phone": deterministic["phone"],
        "skills": deterministic["skills"],

        # NER fields
        "locations": locations,
        "designations": designations,
        "companies": companies,
        "colleges": colleges,
        "degrees": degrees,

        # Existing section extraction
        "education": deterministic["education"],
        "experience": deterministic["experience"],
        "projects": deterministic["projects"],
        "certifications": deterministic["certifications"],
    }