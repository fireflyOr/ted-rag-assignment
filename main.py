from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "SUCCESS", "message": "The server is running!"}
