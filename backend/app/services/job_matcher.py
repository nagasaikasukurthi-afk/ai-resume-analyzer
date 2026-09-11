import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SEMANTIC MODEL
# ============================================================

try:
    semantic_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True
    )
except Exception:
    semantic_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# SKILL LIBRARY
# ============================================================

SKILLS = [
    # Programming
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
    "Go", "Golang", "Rust", "Kotlin", "Swift", "PHP", "Ruby",

    # Web
    "HTML", "CSS", "React", "React.js", "Angular", "Vue.js",
    "Next.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI",
    "Spring", "Spring Boot",

    # APIs
    "REST API", "REST APIs", "GraphQL", "Microservices",

    # Databases
    "SQL", "MySQL", "PostgreSQL", "SQLite", "MongoDB",
    "Oracle", "Redis", "Cassandra",

    # Data
    "Pandas", "NumPy", "Matplotlib", "Seaborn",
    "Jupyter", "Excel", "Power BI", "Tableau",

    # Machine Learning / AI
    "Machine Learning", "Deep Learning", "NLP",
    "Natural Language Processing", "Artificial Intelligence", "AI",
    "Scikit-learn", "TensorFlow", "PyTorch",
    "Transformers", "Generative AI", "LLM", "Large Language Models",

    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Google Cloud",
    "Docker", "Kubernetes", "Jenkins",
    "Terraform", "Ansible", "Linux",
    "CI/CD", "DevOps", "DevSecOps",

    # Development Tools
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Postman", "VS Code",

    # Software Engineering
    "Data Structures", "Algorithms",
    "OOP", "Object-Oriented Programming",
    "DBMS", "System Design",
    "Unit Testing", "PyTest",
    "Agile", "Scrum",

    # Project / Business
    "Project Management", "Product Management",
    "Business Analysis", "Requirements Analysis",
    "Documentation", "Technical Documentation",

    # Marketing
    "Digital Marketing", "SEO", "SEM",
    "Content Marketing", "Social Media Marketing",
    "Google Analytics",

    # Finance / Accounting
    "Accounting", "Financial Analysis", "Financial Modeling",
    "Bookkeeping", "Taxation", "Auditing",

    # HR
    "Recruitment", "Talent Acquisition",
    "Human Resources", "Payroll",
    "Employee Relations",

    # Engineering / Design
    "AutoCAD", "SolidWorks", "MATLAB",
    "CAD", "Mechanical Design",
    "Electrical Engineering",

    # Healthcare / Research
    "Clinical Research", "Data Analysis",
    "Research", "Statistics",

    # General Professional
    "Communication", "Leadership", "Teamwork",
    "Problem Solving", "Critical Thinking",
    "Time Management", "Presentation",
    "Negotiation", "Stakeholder Management",
]


# ============================================================
# SOFT SKILLS
# ============================================================

SOFT_SKILLS = {
    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "critical thinking",
    "time management",
    "presentation",
    "negotiation",
    "stakeholder management",
}


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node.js",
    "nodejs": "node.js",
    "nextjs": "next.js",
    "vuejs": "vue.js",
    "golang": "go",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "rest api": "rest api",
    "rest apis": "rest api",
    "artificial intelligence": "ai",
    "machine learning": "machine learning",
    "natural language processing": "nlp",
    "object oriented programming": "oop",
}


# ============================================================
# NORMALIZE SKILL
# ============================================================

def normalize_skill(skill: str) -> str:

    skill = skill.lower().strip()

    skill = re.sub(
        r"[^a-z0-9+#./ -]",
        "",
        skill
    )

    return SKILL_ALIASES.get(
        skill,
        skill
    )


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills_from_text(text: str):

    if not text:
        return []

    found_skills = []

    sorted_skills = sorted(
        SKILLS,
        key=len,
        reverse=True
    )

    for skill in sorted_skills:

        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            found_skills.append(skill)

    unique_skills = {}

    for skill in found_skills:

        normalized = normalize_skill(skill)

        if normalized not in unique_skills:

            unique_skills[normalized] = skill

    return list(
        unique_skills.values()
    )


# ============================================================
# CLEAN SECTION
# ============================================================

def clean_section_text(text: str):

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^[•●▪◦\-*]\s*",
            "",
            line
        )

        lines.append(line)

    return " ".join(lines)


# ============================================================
# SECTION HEADINGS
# ============================================================

SKILL_HEADINGS = [
    "skills",
    "technical skills",
    "required skills",
    "key skills",
    "core skills",
    "preferred skills",
]


RESPONSIBILITY_HEADINGS = [
    "responsibilities",
    "job responsibilities",
    "key responsibilities",
    "your responsibilities",
    "role overview",
    "what you'll do",
    "what you will do",
]


QUALIFICATION_HEADINGS = [
    "qualifications",
    "qualification",
    "requirements",
    "education",
    "educational qualification",
    "educational qualifications",
]


EXPERIENCE_HEADINGS = [
    "experience",
    "work experience",
    "professional experience",
    "required experience",
    "experience requirements",
]


GENERAL_STOP_HEADINGS = [
    "skills",
    "technical skills",
    "required skills",
    "preferred skills",
    "responsibilities",
    "job responsibilities",
    "key responsibilities",
    "qualifications",
    "qualification",
    "requirements",
    "education",
    "educational qualification",
    "experience",
    "work experience",
    "professional experience",
    "about",
    "benefits",
    "perks",
    "salary",
    "location",
]


# ============================================================
# EXTRACT SECTION BY HEADINGS
# ============================================================

def extract_section_by_headings(
    text: str,
    headings,
    stop_headings
):

    if not text:
        return ""

    # --------------------------------------------------------
    # Normalize common inline JD formatting
    # --------------------------------------------------------

    normalized_text = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    # --------------------------------------------------------
    # Heading patterns
    # --------------------------------------------------------

    heading_pattern = "|".join(
        re.escape(h)
        for h in sorted(
            headings,
            key=len,
            reverse=True
        )
    )

    stop_pattern = "|".join(
        re.escape(h)
        for h in sorted(
            stop_headings,
            key=len,
            reverse=True
        )
    )

    # --------------------------------------------------------
    # Case 1:
    # Heading: content
    #
    # Example:
    # Responsibilities: Develop websites...
    # Qualifications: B.Tech...
    # --------------------------------------------------------

    inline_pattern = rf"""
        (?:^|\s)
        (?:{heading_pattern})
        \s*:\s*
        (.*?)
        (?=
            \s+(?:{stop_pattern})
            \s*:
            |\Z
        )
    """

    match = re.search(
        inline_pattern,
        normalized_text,
        re.IGNORECASE |
        re.DOTALL |
        re.VERBOSE
    )

    if match:

        return clean_section_text(
            match.group(1)
        )

    # --------------------------------------------------------
    # Case 2:
    # Standalone heading
    #
    # Example:
    #
    # Responsibilities:
    # Develop websites...
    # Qualifications:
    # B.Tech...
    # --------------------------------------------------------

    multiline_heading_pattern = "|".join(
        re.escape(h)
        for h in sorted(
            headings,
            key=len,
            reverse=True
        )
    )

    multiline_stop_pattern = "|".join(
        re.escape(h)
        for h in sorted(
            stop_headings,
            key=len,
            reverse=True
        )
    )

    pattern = rf"""
        ^\s*
        (?:{multiline_heading_pattern})
        \s*:?
        \s*
        (.*?)
        (?=
            ^\s*
            (?:{multiline_stop_pattern})
            \s*:?
            \s*$
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

        return clean_section_text(
            match.group(1)
        )

    return ""

# ============================================================
# JOB REQUIREMENTS
# ============================================================

def extract_job_requirements(
    job_description: str
):

    text = job_description.strip()

    skills_section = extract_section_by_headings(
        text,
        SKILL_HEADINGS,
        GENERAL_STOP_HEADINGS
    )

    responsibilities = extract_section_by_headings(
        text,
        RESPONSIBILITY_HEADINGS,
        GENERAL_STOP_HEADINGS
    )

    qualifications = extract_section_by_headings(
        text,
        QUALIFICATION_HEADINGS,
        GENERAL_STOP_HEADINGS
    )

    experience_requirements = extract_section_by_headings(
        text,
        EXPERIENCE_HEADINGS,
        GENERAL_STOP_HEADINGS
    )
        # --------------------------------------------------------
    # Qualification fallback for inline JDs
    # --------------------------------------------------------
    if not qualifications.strip():

        qualification_patterns = [
            r"\bB\.?\s*Tech\b",
            r"\bBTech\b",
            r"\bB\.?\s*E\.?\b",
            r"\bBachelor\s+of\s+Technology\b",
            r"\bBachelor\s+of\s+Engineering\b",
            r"\bB\.?\s*Sc\b",
            r"\bBSc\b",
            r"\bBachelor\s+of\s+Science\b",
            r"\bBCA\b",
            r"\bBachelor\s+of\s+Computer\s+Applications\b",
            r"\bM\.?\s*Tech\b",
            r"\bMTech\b",
            r"\bMaster\s+of\s+Technology\b",
            r"\bMCA\b",
            r"\bMaster\s+of\s+Computer\s+Applications\b",
            r"\bM\.?\s*Sc\b",
            r"\bMSc\b",
            r"\bMaster\s+of\s+Science\b",
            r"\bComputer\s+Science\b",
            r"\bComputer\s+Science\s+Engineering\b",
            r"\bCSE\b",
            r"\bInformation\s+Technology\b",
            r"\bIT\b",
            r"\bElectrical\s+Engineering\b",
            r"\bEEE\b",
            r"\bMechanical\s+Engineering\b",
            r"\bME\b",
            r"\bElectronics\s+Engineering\b",
            r"\bECE\b",
        ]

        qualification_matches = []

        for pattern in qualification_patterns:
            matches = re.findall(
                pattern,
                text,
                re.IGNORECASE
            )

            qualification_matches.extend(matches)

        if qualification_matches:
            qualifications = " ".join(
                dict.fromkeys(qualification_matches)
            ).strip()

    # --------------------------------------------------------
    # Extract experience requirements from Qualifications
    # or from the full job description
    # --------------------------------------------------------
    if not experience_requirements.strip():

        qualification_experience_matches = re.findall(
            r"\b\d+\s*(?:\+|[-–]\s*\d+)?\s*years?\b"
            r"(?:\s+of\s+(?:professional\s+)?experience)?"
            r"(?:\s+(?:required|preferred|needed|minimum))?",
            qualifications,
            re.IGNORECASE
        )

        if qualification_experience_matches:
            experience_requirements = " ".join(
                qualification_experience_matches
            ).strip()

    # --------------------------------------------------------
    # Final fallback: search the complete JD
    # --------------------------------------------------------
    if not experience_requirements.strip():

        general_experience_matches = re.findall(
            r"\b\d+\s*(?:\+|[-–]\s*\d+)?\s*years?\b"
            r"(?:\s+of\s+(?:professional\s+)?experience)?"
            r"(?:\s+(?:required|preferred|needed|minimum))?",
            text,
            re.IGNORECASE
        )

        if general_experience_matches:
            experience_requirements = " ".join(
                general_experience_matches
            ).strip()
    # --------------------------------------------------------
    # Extract required + preferred skills
    # --------------------------------------------------------

    required_skills_section = extract_section_by_headings(
        text,
        [
            "skills",
            "technical skills",
            "required skills",
            "key skills",
            "core skills",
        ],
        GENERAL_STOP_HEADINGS
    )

    preferred_skills_section = extract_section_by_headings(
        text,
        [
            "preferred skills",
        ],
        GENERAL_STOP_HEADINGS
    )

    required_skills = extract_skills_from_text(
        required_skills_section
    )

    preferred_skills = extract_skills_from_text(
        preferred_skills_section
    )

    # --------------------------------------------------------
    # Fallback: extract skills from the complete JD
    # --------------------------------------------------------
    if not required_skills and not preferred_skills:
        job_skills = extract_skills_from_text(text)
    else:
        job_skills = list(
            dict.fromkeys(
                required_skills + preferred_skills
            )
        )

    # --------------------------------------------------------
    # Soft skills
    # --------------------------------------------------------

    soft_skills = []

    for skill in job_skills:

        if normalize_skill(skill) in SOFT_SKILLS:

            soft_skills.append(skill)

    technical_skills = [
        skill
        for skill in job_skills
        if normalize_skill(skill)
        not in SOFT_SKILLS
    ]

    # --------------------------------------------------------
    # Fresher detection
    # --------------------------------------------------------

    fresher_patterns = [
        "freshers are welcome",
        "fresher",
        "freshers",
        "entry level",
        "entry-level",
        "0-1 years",
        "0 to 1 years",
        "0 years experience",
        "no experience required",
        "graduates welcome",
        "recent graduates",
    ]

    is_fresher_friendly = any(
        pattern in text.lower()
        for pattern in fresher_patterns
    )

    return {
        "skills": job_skills,
        "soft_skills": soft_skills,
        "technical_skills": technical_skills,
        "responsibilities": responsibilities,
        "qualifications": qualifications,
        "experience_requirements": experience_requirements,
        "is_fresher_friendly": is_fresher_friendly,
    }

# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def calculate_similarity(
    resume_text: str,
    job_description: str
):

    if not resume_text.strip() or not job_description.strip():

        return 0.0

    resume_embedding = semantic_model.encode(
        [resume_text],
        normalize_embeddings=True
    )

    job_embedding = semantic_model.encode(
        [job_description],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    return round(
        float(similarity * 100),
        2
    )


# ============================================================
# SKILL MATCH
# ============================================================

def calculate_skill_match(
    resume_text: str,
    job_skills
):

    resume_skills = extract_skills_from_text(
        resume_text
    )

    resume_normalized = {
        normalize_skill(skill)
        for skill in resume_skills
    }

    job_normalized = {
        normalize_skill(skill)
        for skill in job_skills
    }

    matched_skills = []
    missing_skills = []

    for skill in job_skills:

        normalized = normalize_skill(skill)

        if normalized in resume_normalized:

            matched_skills.append(skill)

        else:

            missing_skills.append(skill)

    if job_normalized:

        score = (
            len(
                job_normalized.intersection(
                    resume_normalized
                )
            )
            /
            len(job_normalized)
        ) * 100

    else:

        score = 0.0

    return {
        "resume_skills": resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "score": round(score, 2),
    }


# ============================================================
# RESUME SECTION
# ============================================================

def extract_resume_section(
    resume_text: str,
    section_names
):

    heading_pattern = "|".join(
        re.escape(section)
        for section in section_names
    )

    stop_sections = (
        r"PROFESSIONAL SUMMARY|"
        r"SUMMARY|"
        r"OBJECTIVE|"
        r"TECHNICAL SKILLS|"
        r"SKILLS|"
        r"EDUCATION|"
        r"EXPERIENCE|"
        r"WORK EXPERIENCE|"
        r"PROFESSIONAL EXPERIENCE|"
        r"PROJECTS|"
        r"PROJECT EXPERIENCE|"
        r"CERTIFICATIONS|"
        r"CERTIFICATES|"
        r"ACHIEVEMENTS|"
        r"LANGUAGES|"
        r"CONTACT"
    )

    pattern = rf"""
        ^\s*
        (?:{heading_pattern})
        \s*:?
        \s*
        (.*?)
        (?=
            ^\s*(?:{stop_sections})\s*:?
            \s*$
            |\Z
        )
    """

    match = re.search(
        pattern,
        resume_text,
        re.IGNORECASE |
        re.MULTILINE |
        re.DOTALL |
        re.VERBOSE
    )

    if match:

        return match.group(1).strip()

    return ""


# ============================================================
# PROJECT TEXT
# ============================================================

def extract_project_text(
    resume_text: str
):

    return extract_resume_section(
        resume_text,
        [
            "PROJECTS",
            "PROJECT EXPERIENCE"
        ]
    )

# --------------------------------------------------------
# Common project title indicators
# --------------------------------------------------------

def extract_individual_projects(
    resume_text: str
):
    """
    Extract individual projects from the PROJECTS section.

    Supports:
    - Project title + technologies on the same line
    - Project title on its own line
    - Numbered project titles
    - PDF bullet points
    - PDF line wrapping
    """

    projects_section = extract_project_text(resume_text)

    if not projects_section.strip():
        return []

    # Fix PDF line wrapping
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        projects_section
    )

    # Normalize lines
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    # Description starters
    description_starters = (
        "developed",
        "develop",
        "built",
        "build",
        "created",
        "create",
        "implemented",
        "implement",
        "designed",
        "design",
        "performed",
        "analyzed",
        "trained",
        "integrated",
        "deployed",
        "developing",
        "using",
        "responsible",
    )

    # Technologies
    technology_pattern = (
        r"Python|Java|JavaScript|TypeScript|C\+\+|C#|"
        r"React(?:\.js)?|Angular|Vue(?:\.js)?|Django|DRF|"
        r"FastAPI|Flask|Node(?:\.js)?|Express(?:\.js)?|"
        r"Next(?:\.js)?|PostgreSQL|MySQL|SQLite|MongoDB|"
        r"AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|"
        r"Pandas|NumPy|Scikit-learn|TensorFlow|PyTorch|"
        r"NLP|HTML|CSS|Git|GitHub|SQL|REST APIs?|"
        r"Machine Learning|Deep Learning"
    )

    invalid_titles = {
        "projects",
        "project experience",
        "professional projects",
        "academic projects",
        "technical projects",
        "personal projects",
    }

    projects = []
    current_project = None

    # Process each line
    for line in lines:

        clean_line = re.sub(
            r"^[•●▪◦\-*]\s*",
            "",
            line
        ).strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # Check for numbered project
        numbered_title = re.match(
            r"^\d+[\.)]\s*(.+)$",
            clean_line
        )

        # Check whether line looks like description
        starts_like_description = lower_line.startswith(
            description_starters
        )

        # Detect technology stack after title
        technology_match = re.search(
            rf"\s+(?=(?:{technology_pattern})\b)",
            clean_line,
            re.IGNORECASE
        )

        # Basic title detection
        word_count = len(clean_line.split())

        looks_like_title = (
            word_count <= 12
            and not clean_line.endswith(".")
            and not starts_like_description
        )

        project_title = clean_line

        # Numbered project
        if numbered_title:

            looks_like_title = True

            project_title = numbered_title.group(1).strip()

        # Project title + technologies
        elif (
            technology_match
            and not starts_like_description
        ):

            possible_title = clean_line[
                :technology_match.start()
            ].strip(" ,:-")

            if (
                possible_title
                and len(possible_title.split()) <= 12
                and possible_title.lower() not in invalid_titles
            ):

                looks_like_title = True

                project_title = possible_title

        # New project
        if looks_like_title:

            title = project_title.strip()

            if title.lower() in invalid_titles:
                continue

            # Save previous project
            if current_project is not None:

                current_project["text"] = re.sub(
                    r"\s+",
                    " ",
                    current_project["text"]
                ).strip()

                projects.append(current_project)

            # Start new project
            current_project = {
                "name": title,
                "text": clean_line
            }

        # Project description
        elif current_project is not None:

            current_project["text"] += (
                " " + clean_line
            )

        # Fallback
        else:

            current_project = {
                "name": clean_line,
                "text": clean_line
            }

    # Save final project
    if current_project is not None:

        current_project["text"] = re.sub(
            r"\s+",
            " ",
            current_project["text"]
        ).strip()

        projects.append(current_project)

    return projects
# ============================================================
# PROFESSIONAL EXPERIENCE
# ============================================================

def resume_has_professional_experience(
    resume_text: str
):

    experience = extract_resume_section(
        resume_text,
        [
            "EXPERIENCE",
            "WORK EXPERIENCE",
            "PROFESSIONAL EXPERIENCE"
        ]
    )

    return bool(
        experience.strip()
    )


# ============================================================
# REQUIRED YEARS
# ============================================================

def extract_required_years(experience_text: str) -> int:
    """
    Extract the minimum/upper required years of professional experience
    from a job description.

    Examples:
    - "3+ years" -> 3
    - "1-2 years" -> 2
    - "5 years of experience" -> 5
    - "freshers welcome" -> 0
    """

    if not experience_text:
        return 0

    text = experience_text.lower()

    # --------------------------------------------------------
    # Explicit fresher / no-experience indicators
    # --------------------------------------------------------
    fresher_patterns = [
        r"\bfreshers?\b",
        r"\bentry[-\s]?level\b",
        r"\brecent graduate\b",
        r"\b0\s*[-–]\s*1\s*years?\b",
        r"\b0\s+to\s+1\s*years?\b",
        r"\bno experience required\b",
        r"\bexperience not required\b",
        r"\bwithout experience\b",
    ]

    for pattern in fresher_patterns:
        if re.search(pattern, text):
            return 0

    # --------------------------------------------------------
    # Experience ranges
    # Example: "1-2 years" -> 2
    # --------------------------------------------------------
    range_matches = re.findall(
        r"(\d+)\s*[-–]\s*(\d+)\s*years?",
        text
    )

    if range_matches:
        return max(
            int(high)
            for low, high in range_matches
        )

    # --------------------------------------------------------
    # Plus years
    # Example: "3+ years" -> 3
    # --------------------------------------------------------
    plus_matches = re.findall(
        r"(\d+)\s*\+\s*years?",
        text
    )

    # --------------------------------------------------------
    # Normal experience statements
    # --------------------------------------------------------
    normal_patterns = [
        r"(?:minimum|at least|required)\s*(?:of\s*)?(\d+)\s*years?",
        r"(\d+)\s*years?\s*(?:of\s*)?(?:professional\s*)?experience",
        r"(\d+)\s*years?\s*experience",
    ]

    years_found = [
        int(year)
        for year in plus_matches
    ]

    for pattern in normal_patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                years_found.append(int(match))
            except (ValueError, TypeError):
                continue

    if not years_found:
        return 0

    return max(years_found)

def analyze_experience_requirement(
    resume_text: str,
    experience_requirements: str,
    fresher_friendly: bool
):
    """
    Analyze whether the candidate's professional experience matches
    the job's experience requirements.
    """

    required_years = extract_required_years(experience_requirements)

    has_professional_experience = resume_has_professional_experience(
        resume_text
    )

    # Fresher-friendly job
    if fresher_friendly and required_years == 0:
        if has_professional_experience:
            return {
                "score": 100.0,
                "required_years": 0,
                "resume_has_experience": True,
                "status": "qualified",
                "message": "Job is fresher-friendly and the resume contains professional experience."
            }

        return {
            "score": 100.0,
            "required_years": 0,
            "resume_has_experience": False,
            "status": "qualified",
            "message": "Job is fresher-friendly. Professional experience is not required."
        }

    # No clear experience requirement
    if required_years == 0:
        return {
            "score": 70.0 if has_professional_experience else 50.0,
            "required_years": 0,
            "resume_has_experience": has_professional_experience,
            "status": "not_specified",
            "message": (
                "The job description does not specify a clear experience requirement."
            )
        }

    # Job requires experience but resume has no professional experience
    if not has_professional_experience:
        return {
            "score": 0.0,
            "required_years": required_years,
            "resume_has_experience": False,
            "status": "gap",
            "message": (
                f"The job requires approximately {required_years} "
                "year(s) of professional experience, but no professional "
                "experience was detected in the resume."
            )
        }

    # Professional experience exists, but exact duration is not available
    return {
        "score": 70.0,
        "required_years": required_years,
        "resume_has_experience": True,
        "status": "partial",
        "message": (
            f"The resume contains professional experience and the job "
            f"requires approximately {required_years} year(s)."
        )
    }

# ============================================================
# EXPERIENCE / PROJECT MATCH
# ============================================================

def calculate_experience_match(
    resume_text: str,
    responsibilities: str
):

    if not responsibilities.strip():

        return 50.0

    resume_experience = extract_resume_section(
        resume_text,
        [
            "EXPERIENCE",
            "WORK EXPERIENCE",
            "PROFESSIONAL EXPERIENCE"
        ]
    )

    projects = extract_project_text(
        resume_text
    )

    relevant_text = (
        resume_experience
        +
        "\n"
        +
        projects
    ).strip()

    if not relevant_text:

        return 10.0

    similarity = calculate_similarity(
        relevant_text,
        responsibilities
    )

    # Fresher project bonus
    if not resume_experience.strip() and projects.strip():

        similarity = min(
            similarity + 10,
            100
        )

    return round(
        similarity,
        2
    )

# ============================================================
# PROJECT RELEVANCE V2
# ============================================================

def calculate_project_relevance(
    resume_text: str,
    job_description: str,
    technical_job_skills: list
):
    """
    Calculate how relevant the candidate's projects are to the job.

    Project relevance combines:
    - Semantic similarity between the project and JD
    - Project-specific technical skill overlap

    The skill score is calculated against the skills that are
    actually relevant to each project, rather than dividing
    by every skill in the complete JD.
    """

    projects = extract_individual_projects(resume_text)

    if not projects:
        return 0.0, []

    relevant_projects = []

    for project in projects:

        project_text = project.get(
            "text",
            ""
        )

        project_name = project.get(
            "name",
            ""
        )

        if not project_text.strip():
            continue

        # ----------------------------------------------------
        # Semantic similarity
        # ----------------------------------------------------

        semantic_similarity = calculate_similarity(
            project_text,
            job_description
        )

        semantic_score = semantic_similarity

        # ----------------------------------------------------
        # Extract skills from project
        # ----------------------------------------------------

        project_skills = extract_skills_from_text(
            project_text
        )

        # ----------------------------------------------------
        # Find technical skills matched by project
        # ----------------------------------------------------

        matched_project_skills = []

        for skill in technical_job_skills:

            normalized_job_skill = normalize_skill(
                skill
            )

            for project_skill in project_skills:

                normalized_project_skill = normalize_skill(
                    project_skill
                )

                if (
                    normalized_job_skill
                    == normalized_project_skill
                ):

                    matched_project_skills.append(
                        skill
                    )

                    break

        # Remove duplicates
        matched_project_skills = list(
            dict.fromkeys(
                matched_project_skills
            )
        )

        # ----------------------------------------------------
        # Project-specific skill score
        # ----------------------------------------------------

        if technical_job_skills:

            skill_score = (
                len(matched_project_skills)
                / len(technical_job_skills)
            ) * 100

        else:

            skill_score = 0.0

        # ----------------------------------------------------
        # Combined project relevance
        # ----------------------------------------------------

        relevance_score = (
            semantic_score * 0.60
            + skill_score * 0.40
        )

        relevant_projects.append(
            {
                "project_name": project_name,
                "relevance_score": round(
                    relevance_score,
                    2
                ),
                "semantic_score": round(
                    semantic_score,
                    2
                ),
                "skill_score": round(
                    skill_score,
                    2
                ),
                "matched_skills": matched_project_skills
            }
        )

    # --------------------------------------------------------
    # Sort highest relevance first
    # --------------------------------------------------------

    relevant_projects.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Overall project relevance
    # --------------------------------------------------------

    if relevant_projects:

        top_projects = relevant_projects[:3]

        # Give more importance to the most relevant projects.
        # Best project: 60%
        # Second project: 30%
        # Third project: 10%

        weights = [0.60, 0.30, 0.10]

        weighted_score = 0.0
        total_weight = 0.0

        for index, project in enumerate(top_projects):

            weight = weights[index]

            weighted_score += (
                project["relevance_score"] * weight
            )

            total_weight += weight

        project_score = (
            weighted_score / total_weight
        )

    else:

        project_score = 0.0

    return round(project_score, 2), relevant_projects    
# ============================================================
# QUALIFICATION MATCH
# ============================================================

def calculate_qualification_match(
    resume_text: str,
    qualifications: str
):

    if not qualifications.strip():

        return {
            "score": 50.0,
            "matched": [],
            "missing": [],
        }

    resume_lower = resume_text.lower()
    qualification_lower = qualifications.lower()

    matched = []
    missing = []

    # --------------------------------------------------------
    # Degree alternatives
    # --------------------------------------------------------

    degree_patterns = {
        "B.Tech / B.E": [
            r"\bb\.?\s*tech\b",
            r"\bbtech\b",
            r"\bb\.?\s*e\b",
            r"\bbe\b",
            r"\bbachelor\s+of\s+technology\b",
            r"\bbachelor\s+of\s+engineering\b",
        ],
        

        "B.Sc": [
            r"\bb\.?\s*sc\b",
            r"\bbsc\b",
            r"\bbachelor\s+of\s+science\b",
        ],

        "BCA": [
            r"\bbca\b",
            r"\bbachelor\s+of\s+computer\s+applications\b",
        ],

        "M.Tech": [
            r"\bm\.?\s*tech\b",
            r"\bmtech\b",
            r"\bmaster\s+of\s+technology\b",
        ],

        "MCA": [
            r"\bmca\b",
            r"\bmaster\s+of\s+computer\s+applications\b",
        ],

        "M.Sc": [
            r"\bm\.?\s*sc\b",
            r"\bmsc\b",
            r"\bmaster\s+of\s+science\b",
        ],
    }

    degree_requirement_found = False

    for display_name, patterns in degree_patterns.items():

        required = any(
            re.search(
                pattern,
                qualification_lower
            )
            for pattern in patterns
        )

        if not required:
            continue

        degree_requirement_found = True

        candidate_has_degree = any(
            re.search(
                pattern,
                resume_lower
            )
            for pattern in patterns
        )

        if candidate_has_degree:

            matched.append(
                display_name
            )

        else:

            missing.append(
                display_name
            )

    # --------------------------------------------------------
    # Field requirements
    # --------------------------------------------------------

    field_patterns = {

        "Computer Science": [
            r"\bcomputer\s+science\b",
            r"\bcomputer\s+science\s+engineering\b",
            r"\bcse\b",
        ],

        "Information Technology": [
            r"\binformation\s+technology\b",
            r"\bi\.?\s*t\.?\b",
        ],

        "Electrical Engineering": [
            r"\belectrical\s+engineering\b",
            r"\beee\b",
        ],

        "Mechanical Engineering": [
            r"\bmechanical\s+engineering\b",
            r"\bme\b",
        ],

        "Electronics Engineering": [
            r"\belectronics\s+engineering\b",
            r"\bece\b",
        ],
    }

    for display_name, patterns in field_patterns.items():

        required = any(
            re.search(
                pattern,
                qualification_lower
            )
            for pattern in patterns
        )

        if not required:
            continue

        candidate_has_field = any(
            re.search(
                pattern,
                resume_lower
            )
            for pattern in patterns
        )

        if candidate_has_field:

            matched.append(
                display_name
            )

        else:

            missing.append(
                display_name
            )

    matched = list(
        dict.fromkeys(matched)
    )

    missing = list(
        dict.fromkeys(missing)
    )

    if (
        not degree_requirement_found
        and not matched
        and not missing
    ):

        return {
            "score": 50.0,
            "matched": [],
            "missing": [],
        }

    total_requirements = (
        len(matched)
        +
        len(missing)
    )

    if total_requirements == 0:

        score = 50.0

    else:

        score = (
            len(matched)
            /
            total_requirements
        ) * 100

    return {
        "score": round(score, 2),
        "matched": matched,
        "missing": missing,
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    missing_skills
):

    recommendations = []

    for skill in missing_skills:

        recommendations.append(
            f"Consider learning or improving {skill}."
        )

    return recommendations


# ============================================================
# FINAL MATCHING ENGINE
# ============================================================

def match_resume_to_job(
    resume_text: str,
    job_description: str
):

    # --------------------------------------------------------
    # Extract job requirements
    # --------------------------------------------------------

    requirements = extract_job_requirements(
        job_description
    )

    job_skills = requirements[
        "skills"
    ]

    soft_skills = requirements[
        "soft_skills"
    ]

    technical_skills = requirements[
        "technical_skills"
    ]

    responsibilities = requirements[
        "responsibilities"
    ]

    qualifications = requirements[
        "qualifications"
    ]

    experience_requirements = requirements[
        "experience_requirements"
    ]

    fresher_friendly = requirements[
        "is_fresher_friendly"
    ]

    # --------------------------------------------------------
    # Skill matching
    # --------------------------------------------------------

    skill_result = calculate_skill_match(
        resume_text,
        technical_skills
    )
    

    skill_score = skill_result[
        "score"
    ]

    # --------------------------------------------------------
    # Soft skill matching
    # --------------------------------------------------------

    resume_skills_normalized = {
        normalize_skill(skill)
        for skill in skill_result[
            "resume_skills"
        ]
    }

    matched_soft_skills = [
        skill
        for skill in soft_skills
        if normalize_skill(skill)
        in resume_skills_normalized
    ]

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_score = calculate_similarity(
        resume_text,
        job_description
    )
    # --------------------------------------------------------
# Project relevance
# --------------------------------------------------------

    project_score, relevant_projects = calculate_project_relevance(
        resume_text,
        job_description,
        technical_skills
)
    
    # --------------------------------------------------------
    # Qualification matching
    # --------------------------------------------------------

    qualification_result = calculate_qualification_match(
        resume_text,
        qualifications
    )

    qualification_score = qualification_result[
        "score"
    ]
        # --------------------------------------------------------
    # Experience / Fresher matching
    # --------------------------------------------------------

    experience_result = analyze_experience_requirement(
        resume_text,
        experience_requirements,
        fresher_friendly
    )

    experience_score = experience_result[
        "score"
    ]

    # --------------------------------------------------------
    # Required experience
    # --------------------------------------------------------

    required_experience_years = extract_required_years(
        experience_requirements
        +
        " "
        +
        job_description
    )

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------
    #
    # Technical skills = 50%
    # Semantic match   = 25%
    # Experience       = 15%
    # Qualification    = 10%
    #
        # --------------------------------------------------------
    # Final match score
    # --------------------------------------------------------

    final_score = (
        skill_score * 0.45
        + semantic_score * 0.20
        + project_score * 0.15
        + qualification_score * 0.10
        + experience_score * 0.10
    )
    

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = generate_recommendations(
        skill_result[
            "missing_skills"
        ]
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "match_score": round(final_score, 2),

        "skill_match_score": skill_score,

        "semantic_similarity": semantic_score,

        "project_relevance_score": project_score,

        "relevant_projects": relevant_projects,

        "qualification_match_score": qualification_score,

        "experience_relevance_score": experience_score,

        "experience_analysis": experience_result,

        "matched_skills": skill_result[
            "matched_skills"
        ],

        "missing_skills": skill_result[
            "missing_skills"
        ],

        "matched_soft_skills": matched_soft_skills,

        "resume_skills": skill_result[
            "resume_skills"
        ],

        "job_skills": job_skills,

        "technical_skills": technical_skills,

        "soft_skills": soft_skills,

        "job_responsibilities": responsibilities,

        "job_qualifications": qualifications,

        "experience_requirements": experience_requirements,

        "required_experience_years": experience_result[
            "required_years"
        ],

        "fresher_friendly": fresher_friendly,

        "qualification_matched": qualification_result[
            "matched"
        ],

        "qualification_missing": qualification_result[
            "missing"
        ],

        "recommendations": generate_recommendations(
            skill_result["missing_skills"]
        )
    }