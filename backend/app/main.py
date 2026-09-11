from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(
    title="AI Resume Analyzer API",
    description="Backend API for resume analysis and job matching",
    version="1.0.0"
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "AI Resume Analyzer API is running",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }