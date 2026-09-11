from app.services.resume_parser import extract_resume_text
from app.services.hybrid_analyzer import analyze_resume_hybrid


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "uploads" / "Naga Sai.pdf"


print("=" * 70)
print("HYBRID RESUME ANALYZER TEST")
print("=" * 70)


# ============================================================
# EXTRACT TEXT
# ============================================================

resume_text = extract_resume_text(PDF_PATH)

print(
    f"\nExtracted characters: {len(resume_text)}"
)


# ============================================================
# HYBRID ANALYSIS
# ============================================================

result = analyze_resume_hybrid(resume_text)


# ============================================================
# FINAL HYBRID RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL HYBRID RESULT")
print("=" * 70)


for key, value in result.items():

    print(f"\n{key.upper()}:")

    if isinstance(value, list):

        if not value:
            print("  None")

        else:

            for item in value:
                print(f"  - {item}")

    else:

        print(value)


print("\n")
print("=" * 70)
print("HYBRID TEST COMPLETE")
print("=" * 70)