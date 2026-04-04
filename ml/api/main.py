from fastapi import FastAPI

app = FastAPI(title="NHRS ML Service", version="v0")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0"}
