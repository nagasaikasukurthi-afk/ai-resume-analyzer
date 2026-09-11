from app.services.ner_extractor import predict_entities


resume_text = """
Sharan Adla
Software Engineer
Bengaluru, Karnataka
Python Django FastAPI PostgreSQL
B.Tech in Computer Science
M. V. G. R College of Engineering
Worked at Infosys Limited
"""


entities = predict_entities(resume_text)


print("=" * 60)
print("NER V4 TEST")
print("=" * 60)

for entity in entities:
    print(
        f"{entity['label']:25} -> {entity['text']}"
    )