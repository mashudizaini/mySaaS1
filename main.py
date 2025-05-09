import os 
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello SaaS!"}

# Gunakan PORT dari environment variable Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)