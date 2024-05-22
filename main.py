import argparse
from fastapi import FastAPI
from transformers import pipeline
import torch
import os
import uvicorn

parser = argparse.ArgumentParser(description="Deploy a sentiment analysis service.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="IP address to bind the server to.")
parser.add_argument("--port", type=int, default=8000, help="Port for the server.")
parser.add_argument("--pipeline_type", type=str, default="text-classification", help="Type of pipeline to use.")

args = parser.parse_args()

app = FastAPI()
modelname = os.getenv("MODELNAME")
tokenizer = os.getenv("TOKENIZER")

class BertModel:
    def __init__(self, pipeline_type, modelname, tokenizer):
        self.classifier = pipeline(
            pipeline_type,
            model=modelname,
            tokenizer=tokenizer,
            framework="pt",
            device=torch.device("cuda:0")
        )

    def classify(self, sentence: str):
        return self.classifier(sentence)

model = BertModel(args.pipeline_type, modelname, tokenizer)

@app.post("/classify")
async def classify(sentence: str):
    return model.classify(sentence)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
