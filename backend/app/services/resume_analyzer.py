import re


SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C",
    "C++",
    "C#",
    "SQL",
    "HTML",
    "CSS",
    "React",
    "React.js",
    "Django",
    "Django REST Framework",
    "FastAPI",
    "Flask",
    "Node.js",
    "Express.js",
    "PostgreSQL",
    "MySQL",
    "SQLite",
    "MongoDB",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "Azure",
    "GCP",
    "Linux",
    "REST API",
    "REST APIs",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Natural Language Processing",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Matplotlib",
    "Seaborn",
    "Jupyter",
    "Data Structures",
    "Algorithms",
    "OOP",
    "Object-Oriented Programming",
    "DBMS",
    "Postman",
]


def extract_email(text: str):
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    match = re.search(pattern, text)

    return match.group(0) if match else None


def extract_phone(text: str):
    pattern = r"(?:\+91[\s-]?)?[6-9]\d{9}"

    match = re.search(pattern, text)

    return match.group(0) if match else None


def extract_skills(text: str):
    found_skills = []

    sorted_skills = sorted(
        SKILLS,
        key=len,
        reverse=True
    )

    for skill in sorted_skills:

        escaped_skill = re.escape(skill)

        pattern = rf"(?<!\w){escaped_skill}(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)

    return list(dict.fromkeys(found_skills))


def extract_section(text: str, section_names):

    heading_pattern = "|".join(
        re.escape(section)
        for section in section_names
    )

    pattern = rf"""
        ^\s*(?:{heading_pattern})\s*:?\s*$
        (.*?)
        (?=
            ^\s*(?:
                PROFESSIONAL SUMMARY|
                SUMMARY|
                OBJECTIVE|
                TECHNICAL SKILLS|
                SKILLS|
                EDUCATION|
                EXPERIENCE|
                WORK EXPERIENCE|
                PROFESSIONAL EXPERIENCE|
                PROJECTS|
                PROJECT EXPERIENCE|
                CERTIFICATIONS|
                CERTIFICATES|
                ACHIEVEMENTS|
                LANGUAGES|
                CONTACT
            )\s*:?\s*$
            |\Z
        )
    """

    match = re.search(
        pattern,
        text,
        re.IGNORECASE |
        re.MULTILINE |
        re.DOTALL |
        re.VERBOSE
    )

    if match:
        return match.group(1).strip()

    return None


def analyze_resume(text: str):

    return {
        "email": extract_email(text),

        "phone": extract_phone(text),

        "skills": extract_skills(text),

        "education": extract_section(
            text,
            ["EDUCATION"]
        ),

        "experience": extract_section(
            text,
            [
                "EXPERIENCE",
                "WORK EXPERIENCE",
                "PROFESSIONAL EXPERIENCE"
            ]
        ),

        "projects": extract_section(
            text,
            [
                "PROJECTS",
                "PROJECT EXPERIENCE"
            ]
        ),

        "certifications": extract_section(
            text,
            [
                "CERTIFICATIONS",
                "CERTIFICATES"
            ]
        ),
    }