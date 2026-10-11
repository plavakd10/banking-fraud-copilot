from fastapi import FastAPI

app = FastAPI (
    title = "Banking Fraud Detection and Investigation Copilot",
    description = "Banking risk-scoring API",
    version = "0.1.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "banking-fraud-copilot"
    }