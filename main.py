# For deployment on FastAPI
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
import torch
import torch.nn as nn
from model import DeBERTaLSTMClassifier


app = FastAPI()

model_path = hf_hub_download(repo_id="khoa-done/phishing-detector", filename="deberta_lstm_checkpoint.pt")
model = DeBERTaLSTMClassifier()
model.load_state_dict(model_path['model_state_dict'])

model.eval()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    try:
        input_tensor = torch.tensor([data.features])
        with torch.no_grad():
            output = model(input_tensor)
        prediction = output.tolist()[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))