from app.services.nlp_matcher import calculate_semantic_similarity


resume = """
Python developer with experience building REST APIs using Django
and FastAPI. Worked with PostgreSQL, Docker and AWS.
"""


job_description = """
We are looking for a backend developer who can develop
scalable API services using Python and modern web frameworks.
Experience with cloud platforms and relational databases is preferred.
"""


score = calculate_semantic_similarity(
    resume,
    job_description
)


print("Resume-JD Semantic Similarity:", score)