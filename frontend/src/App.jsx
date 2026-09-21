import { useState } from "react";
import "./App.css";

function App() {
  const [resume, setResume] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const sampleJobDescription = `We are looking for a Python Full Stack Developer.

Responsibilities:
- Develop web applications using Python and React.js.
- Build REST APIs using FastAPI or Django.
- Work with MySQL and MongoDB databases.
- Develop responsive frontend applications.
- Write clean and maintainable code.
- Debug and troubleshoot application issues.

Required Skills:
- Python
- JavaScript
- React.js
- HTML
- CSS
- REST APIs
- MySQL
- MongoDB
- Git
- GitHub
- AWS

Good communication and problem-solving skills are preferred.`;
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleResumeChange = (event) => {
    const file = event.target.files[0];

    if (file) {
      setResume(file);
      setError("");
      setResult(null);
    }
  };
  const handleLoadSample = () => {
  setJobDescription(sampleJobDescription);
  setError("");
};

const handleClearJobDescription = () => {
  setJobDescription("");
};
  const handleReset = () => {
  setResume(null);
  setJobDescription("");
  setResult(null);
  setError("");

  const fileInput = document.getElementById("resume-upload");

  if (fileInput) {
    fileInput.value = "";
  }

  window.scrollTo({
    top: document.getElementById("analyzer")?.offsetTop || 0,
    behavior: "smooth",
  });
};
const handleAnalyze = async () => {
    if (!resume) {
      setError("Please upload your resume first.");
      return;
    }

    if (!jobDescription.trim()) {
      setError("Please enter the job description.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append("file", resume);
    formData.append("job_description", jobDescription);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/resume/match",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Resume analysis failed."
        );
      }

      setResult(data);

    } catch (error) {
      console.error(error);

      setError(
        error.message ||
        "Unable to connect to the backend."
      );

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* Navbar */}

      <header className="navbar">

        <div className="logo">
          AI Resume Analyzer
        </div>

        <nav>
          <a href="#home">Home</a>
          <a href="#analyzer">Analyzer</a>
          <a href="#about">About</a>
        </nav>

      </header>


      {/* Hero */}

      <main id="home">

        <section className="hero">

          <div className="hero-content">

            <p className="badge">
              AI-Powered Resume Analysis
            </p>

            <h1>
              Make Your Resume
              <span> Job-Ready</span>
            </h1>

            <p className="hero-text">
              Upload your resume and compare it with a
              job description to discover your ATS match
              score, missing skills, and areas for
              improvement.
            </p>

            <a
              href="#analyzer"
              className="hero-button"
            >
              Analyze My Resume
            </a>

          </div>

        </section>


        {/* Analyzer */}

        <section
          id="analyzer"
          className="analyzer-section"
        >

          <div className="section-heading">

            <p className="section-label">
              RESUME ANALYZER
            </p>

            <h2>
              Analyze Your Resume
            </h2>

            <p>
              Upload your resume and paste the job
              description below.
            </p>

          </div>


          <div className="analyzer-card">

            {/* Resume Upload */}

            <div className="upload-box">

              <div className="upload-icon">
                📄
              </div>

              <h3>
                Upload Your Resume
              </h3>

              <p>
                PDF or DOCX files are supported
              </p>

              <label
                htmlFor="resume-upload"
                className="upload-button"
              >
                Choose Resume
              </label>

              <input
                id="resume-upload"
                type="file"
                accept=".pdf,.docx"
                onChange={handleResumeChange}
              />

              {resume && (
                <p className="file-name">
                  Selected:
                  {" "}
                  <strong>
                    {resume.name}
                  </strong>
                </p>
              )}

            </div>


            {/* Job Description */}

            <div className="job-description">

              <label htmlFor="job-description">
                Job Description
              </label>

              <textarea
                id="job-description"
                placeholder="Paste the complete job description here..."
                value={jobDescription}
                onChange={(event) =>
                  setJobDescription(
                    event.target.value
                  )
                }
              />

              <p className="character-count">
                {jobDescription.length} characters
              </p>
              <div className="jd-actions">
  <button
    type="button"
    className="sample-jd-button"
    onClick={handleLoadSample}
  >
    ✨ Load Sample JD
  </button>

  <button
    type="button"
    className="clear-jd-button"
    onClick={handleClearJobDescription}
    disabled={!jobDescription}
  >
    Clear
  </button>
</div>

            </div>


            {/* Error */}

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}


            {/* Analyze */}

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >

              {loading ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing Resume...
                </>
            ) : (
              <>
                ✨ Analyze Resume
              </>
            )}


            </button>
            
{loading && (
  <div className="analysis-status">
    <p>🤖 AI is analyzing your resume...</p>

    <span>
      Extracting skills • Comparing job requirements • Calculating match score
    </span>
  </div>
)}


           {/* Results */}

{result && (
  <div className="results">

    <div className="results-header">
      <p className="section-label">ANALYSIS RESULTS</p>
      <h2>Resume Analysis Complete 🎉</h2>
      <p>{result.filename}</p>
    </div>


    {/* Overall Score */}

    <div className="score-card">

      <div className={`score-circle ${result.match_result.match_score >= 70 ? "good" : result.match_result.match_score >= 40 ? "average" : "low"}`}>
        <span>{result.match_result.match_score}%</span>
        <small>Match</small>
      </div>

      <div className="score-details">

        <h3>Overall Resume Match</h3>

        <p>
          Your resume has a{" "}
          <strong>
            {result.match_result.match_score}%
          </strong>{" "}
          match with this job description.
        </p>
        <div className="match-interpretation">
  {result.match_result.match_score >= 70 ? (
    <>
      <strong>Excellent Match</strong>
      <span>Your resume strongly aligns with this job description.</span>
    </>
  ) : result.match_result.match_score >= 60 ? (
    <>
      <strong>Good Match</strong>
      <span>Your resume aligns well with this role, with some areas to improve.</span>
    </>
  ) : result.match_result.match_score >= 40 ? (
    <>
      <strong>Moderate Match</strong>
      <span>Your resume has a reasonable alignment, but several skills can be improved.</span>
    </>
  ) : (
    <>
      <strong>Low Match</strong>
      <span>Your resume needs significant improvement for this role.</span>
    </>
  )}
</div>

      </div>
      <div className="match-summary">

  <h4>📋 Match Summary</h4>

  <p>
    Your resume matches{" "}
    <strong>
      {result.match_result.match_score}%
    </strong>{" "}
    of this job description.
  </p>

  <div className="summary-row">
    <span className="summary-label">
      ✓ Strong Areas
    </span>

    <div className="summary-skills">
      {(result.match_result.matched_skills || [])
        .slice(0, 5)
        .map((skill, index) => (
          <span
            className="summary-tag matched-summary-tag"
            key={index}
          >
            {skill}
          </span>
        ))}
    </div>
  </div>

  <div className="summary-row">
    <span className="summary-label">
      ⚠ Improve
    </span>

    <div className="summary-skills">
      {(result.match_result.missing_skills || [])
        .slice(0, 5)
        .map((skill, index) => (
          <span
            className="summary-tag missing-summary-tag"
            key={index}
          >
            {skill}
          </span>
        ))}
    </div>
  </div>

</div>

    </div>


    {/* Score Breakdown */}

    <div className="score-grid">

<div className="score-item">
  <span>Skill Match</span>

  <strong>
    {result.match_result.skill_match_score}%
  </strong>

  <div className="score-bar">
    <div
      className={`score-bar-fill ${
  result.match_result.skill_match_score >= 70
    ? "good"
    : result.match_result.skill_match_score >= 40
    ? "average"
    : "low"
}`}
      style={{
        width: `${result.match_result.skill_match_score}%`
      }}
    ></div>
  </div>
</div>

<div className="score-item">
  <span>Semantic Similarity</span>

  <strong>
    {result.match_result.semantic_similarity}%
  </strong>

  <div className="score-bar">
    <div
      className={`score-bar-fill ${result.match_result.semantic_similarity >= 70 ? "good" : result.match_result.semantic_similarity >= 40 ? "average" : "low"}`}
      style={{
        width: `${result.match_result.semantic_similarity}%`
      }}
    ></div>
  </div>
</div>

<div className="score-item">
  <span>Project Relevance</span>

  <strong>
    {result.match_result.project_relevance_score}%
  </strong>

  <div className="score-bar">
    <div
      className={`score-bar-fill ${result.match_result.project_relevance_score >= 70 ? "good" : result.match_result.project_relevance_score >= 40 ? "average" : "low"}`}
      style={{
        width: `${result.match_result.project_relevance_score}%`
      }}
    ></div>
  </div>
</div>

      <div className="score-item">
  <span>Qualification Match</span>

  <strong>
    {result.match_result.qualification_match_score}%
  </strong>

  <div className="score-bar">
    <div
      className={`score-bar-fill ${result.match_result.qualification_match_score >= 70 ? "good" : result.match_result.qualification_match_score >= 40 ? "average" : "low"}`}
      style={{
        width: `${result.match_result.qualification_match_score}%`
      }}
    ></div>
  </div>
</div>

        <div className="score-item">
  <span>Experience Relevance</span>

  <strong>
    {result.match_result.experience_relevance_score}%
  </strong>

  <div className="score-bar">
    <div
      className={`score-bar-fill ${result.match_result.experience_relevance_score >= 70 ? "good" : result.match_result.experience_relevance_score >= 40 ? "average" : "low"}`}
      style={{
        width: `${result.match_result.experience_relevance_score}%`
      }}
    ></div>
  </div>
</div>

</div>
    {/* Skills */}

    <div className="skills-grid">

      {/* Matched Skills */}

      <div className="result-category matched-skills">

        <h3>✓ Matched Skills</h3>

        <div className="skill-list">

          {result.match_result.matched_skills.map(
            (skill, index) => (
              <span
                className="skill-tag matched-tag"
                key={index}
              >
                {skill}
              </span>
            )
          )}

        </div>

      </div>


      {/* Missing Skills */}

      <div className="result-category missing-skills">

        <h3>✕ Missing Skills</h3>

        <div className="skill-list">

          {result.match_result.missing_skills.map(
            (skill, index) => (
              <span
                className="skill-tag missing-tag"
                key={index}
              >
                {skill}
              </span>
            )
          )}

        </div>

      </div>

    </div>


    {/* Recommendations */}

    <div className="result-category recommendations">

      <h3>💡 Recommendations</h3>

      <div className="recommendation-list">

        {result.match_result.recommendations.map(
          (recommendation, index) => (
            <div
              className="recommendation-item"
              key={index}
            >
              <span>→</span>
              <p>{recommendation}</p>
            </div>
          )
        )}

      </div>

    </div>


    {/* Relevant Projects */}

    <div className="projects-section">

      <h3>📁 Relevant Projects</h3>

      <div className="projects-grid">

        {result.match_result.relevant_projects.map(
          (project, index) => (
            <div
              className="project-card"
              key={index}
            >

              <h4>
                {project.project_name}
              </h4>

              <div className="project-score">
                Relevance:
                <strong>
                  {project.relevance_score}%
                </strong>
              </div>

              <p>
                Semantic Score:
                {" "}
                {project.semantic_score}%
              </p>

              <p>
                Skill Score:
                {" "}
                {project.skill_score}%
              </p>

              <div className="skill-list">

                {project.matched_skills.map(
                  (skill, skillIndex) => (
                    <span
                      className="skill-tag matched-tag"
                      key={skillIndex}
                    >
                      {skill}
                    </span>
                  )
                )}

              </div>

            </div>
          )
        )}

      </div>

          </div>

      {/* Analyze Another Resume */}

      <div className="reset-analysis">
        <button
          type="button"
          className="reset-button"
          onClick={handleReset}
        >
          ← Analyze Another Resume
        </button>
      </div>

    </div>
)}

          </div>
        </section>

        {/* Features */}

        <section
          id="about"
          className="features-section"
        >
          <div className="section-heading">
            <p className="section-label">
              FEATURES
            </p>

            <h2>
              What You'll Get
            </h2>
          </div>

          <div className="features">

            <div className="feature-card">
              <div className="feature-icon">
                📊
              </div>

              <h3>
                ATS Match Score
              </h3>

              <p>
                Understand how closely your resume
                matches the target job.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                🎯
              </div>

              <h3>
                Skill Matching
              </h3>

              <p>
                Identify important skills from the
                job description that are missing from
                your resume.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">
                💡
              </div>

              <h3>
                Improvement Tips
              </h3>

              <p>
                Get practical suggestions to make
                your resume stronger.
              </p>
            </div>

          </div>
        </section>

      </main>

      <footer>
        <p>
          © 2026 AI Resume Analyzer.
          Built with React & FastAPI.
        </p>
      </footer>

    </div>
  );
}

export default App;