from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.resume_parser import extract_resume_text
from app.services.hybrid_analyzer import analyze_resume_hybrid
from app.services.job_matcher import match_resume_to_job

import os
import shutil


router = APIRouter(prefix="/resume", tags=["Resume"])

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):

    allowed_extensions = [".pdf", ".docx"]

    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_resume_text(file_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract resume text: {str(e)}"
        )

    return {
        "filename": file.filename,
        "message": "Resume uploaded successfully",
        "text": extracted_text
    }

@router.post("/analyze")
async def analyze_uploaded_resume(file: UploadFile = File(...)):

    allowed_extensions = [".pdf", ".docx"]

    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_resume_text(file_path)

        analysis = analyze_resume_hybrid(extracted_text)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Resume analysis failed: {str(e)}"
        )

    return {
        "filename": file.filename,
        "message": "Resume analyzed successfully",
        "analysis": analysis
    }

@router.post("/match")
async def match_resume(
    file: UploadFile = File(...),
    job_description: str = ""
):

    allowed_extensions = [".pdf", ".docx"]

    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        resume_text = extract_resume_text(
            file_path
        )

        match_result = match_resume_to_job(
            resume_text,
            job_description
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Job matching failed: {str(e)}"
        )

    return {
        "filename": file.filename,
        "message": "Resume matched successfully",
        "job_description": job_description,
        "match_result": match_result
    }