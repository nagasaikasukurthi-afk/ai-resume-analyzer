from app.services.resume_parser import extract_resume_text
from app.services.job_matcher import (
    extract_job_requirements,
    calculate_qualification_match,
    extract_required_years,
    match_resume_to_job,
)


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESUME_PATH = BASE_DIR / "uploads" / "Naga Sai.pdf"


def get_resume():
    return extract_resume_text(RESUME_PATH)


def test_inline_skill_extraction():
    jd = """
    Python Full Stack Developer.
    We are looking for candidates with Python, Django, React,
    JavaScript, HTML, CSS, REST APIs, PostgreSQL, Git and GitHub.
    """

    result = extract_job_requirements(jd)

    assert "Python" in result["skills"]
    assert "Django" in result["skills"]
    assert "React" in result["skills"]
    assert "PostgreSQL" in result["skills"]
    assert "GitHub" in result["skills"]


def test_technical_and_soft_skill_separation():
    jd = """
    Python Developer with Python, Django, React and PostgreSQL.
    Good communication, teamwork and problem-solving skills are preferred.
    """

    result = extract_job_requirements(jd)

    assert "Python" in result["technical_skills"]
    assert "Django" in result["technical_skills"]

    assert "Communication" in result["soft_skills"]
    assert "Teamwork" in result["soft_skills"]

    assert "Communication" not in result["technical_skills"]
    assert "Teamwork" not in result["technical_skills"]


def test_qualification_match():
    resume = get_resume()

    jd = "B.Tech or B.E in Computer Science or a related field."

    result = calculate_qualification_match(
        resume,
        jd
    )

    assert result["score"] == 100.0
    assert "B.Tech / B.E" in result["matched"]
    assert "Computer Science" in result["matched"]
    assert result["missing"] == []


def test_qualification_gap():
    resume = get_resume()

    jd = "B.Tech or B.E in Mechanical Engineering."

    result = calculate_qualification_match(
        resume,
        jd
    )

    assert "B.Tech / B.E" in result["matched"]
    assert "Mechanical Engineering" in result["missing"]
    assert result["score"] == 50.0


def test_no_qualification_requirement():
    resume = get_resume()

    jd = "Good communication and teamwork skills are preferred."

    result = calculate_qualification_match(
        resume,
        jd
    )

    assert result["score"] == 50.0
    assert result["matched"] == []
    assert result["missing"] == []


def test_experience_year_extraction():
    assert extract_required_years("3+ years of experience") == 3
    assert extract_required_years("1-2 years of experience") == 2
    assert extract_required_years("5 years of experience") == 5
    assert extract_required_years("Freshers welcome") == 0


def test_full_resume_job_match():
    resume = get_resume()

    jd = """
    Python Full Stack Developer.
    Entry-level position for freshers.

    We are looking for candidates with Python, Django, React,
    JavaScript, HTML, CSS, REST APIs, PostgreSQL, Git and GitHub skills.

    Candidates should have a B.Tech or B.E in Computer Science
    or a related field.

    Freshers are welcome and no professional experience is required.

    Good communication, teamwork and problem-solving skills are preferred.
    """

    result = match_resume_to_job(
        resume,
        jd
    )

    assert result["match_score"] >= 0
    assert result["match_score"] <= 100

    assert result["skill_match_score"] == 80.0

    assert result["qualification_match_score"] == 100.0

    assert result["experience_relevance_score"] == 100.0

    assert "Python" in result["matched_skills"]
    assert "Django" in result["matched_skills"]
    assert "React" in result["matched_skills"]

    assert "HTML" in result["missing_skills"]
    assert "CSS" in result["missing_skills"]

    assert result["qualification_matched"] == [
        "B.Tech / B.E",
        "Computer Science",
    ]

    assert result["qualification_missing"] == []

    assert len(result["recommendations"]) >= 1