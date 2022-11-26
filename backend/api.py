import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Tweet Analyzer App")

@app.get('/')
def root():
    return {'message': 'example'}
