from app.services.job_matcher import match_resume_to_job


resume = """
Python developer with experience in Django, FastAPI, PostgreSQL,
MySQL, REST APIs, Docker, AWS, Git and GitHub.

Built an Employee Management System using Django and PostgreSQL.

Built a Customer Purchase Prediction machine learning project
using Python, Pandas, NumPy and Scikit-learn.

Bachelor of Technology in Computer Science Engineering.
"""


job_description = """
We are looking for a Python Backend Developer.

Skills:
Python, Django, FastAPI, REST APIs, PostgreSQL, Docker, AWS, Git.

Responsibilities:
Develop backend applications and REST APIs using Python.
Build scalable services and work with relational databases.
Deploy applications using cloud technologies.

Qualifications:
Bachelor's degree in Computer Science or related field.
"""


result = match_resume_to_job(
    resume,
    job_description
)


print("\n===== V4 NLP MATCH RESULT =====")

print("Match Score:", result["match_score"])
print("Skill Score:", result["skill_match_score"])
print("Semantic Score:", result["semantic_similarity"])
print("Experience Relevance:", result["experience_relevance_score"])

print("\nMatched Skills:")
print(result["matched_skills"])

print("\nMissing Skills:")
print(result["missing_skills"])