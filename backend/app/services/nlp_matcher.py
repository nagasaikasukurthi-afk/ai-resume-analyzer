from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def calculate_semantic_similarity(
    resume_text: str,
    job_description: str
) -> float:

    if not resume_text.strip() or not job_description.strip():
        return 0.0

    resume_embedding = model.encode(
        [resume_text],
        normalize_embeddings=True
    )

    job_embedding = model.encode(
        [job_description],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    similarity_percentage = float(similarity * 100)

    return round(similarity_percentage, 2)