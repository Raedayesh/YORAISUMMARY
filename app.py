from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class SummarizeRequest(BaseModel):
    inputs: str

@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    summary = summarizer(request.inputs, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}
